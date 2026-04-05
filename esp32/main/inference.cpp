#include "esp_log.h"
#include "esp_attr.h"
#include "model.h"
#include "inference.h"
#include <math.h>

// Include TFLM
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "esp_heap_caps.h"

// Tensor arena size. Increased to 2MB for MobileNetV2-sized models on Sense (8MB PSRAM available)
#define TENSOR_ARENA_SIZE (2048 * 1024)

// Static variables
static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static uint8_t *tensor_arena = nullptr; // Allocated in PSRAM
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static const char *TAG_INF = "Inference";

static float input_scale = 1.0f;
static int input_zero_point = 0;

/**
 * @brief Initialize the TFLite Micro interpreter with the model.
 * 
 * @return True if initialization was successful, false otherwise.
 */
bool inference_init()
{
    // Allocate arena on PSRAM
    if (tensor_arena == nullptr) {
        tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (tensor_arena == nullptr) {
            ESP_LOGE(TAG_INF, "Failed to allocate tensor arena in PSRAM!");
            return false;
        }
    }


    // Load TFlite model
    model = tflite::GetModel(model_binary);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG_INF, "Model schema mismatch!");
        return false;
    }

    // Create an interpreter using MicroMutableOpResolver
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddAdd();
    resolver.AddAveragePool2D();
    resolver.AddMean();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddFullyConnected();
    resolver.AddDequantize();
    resolver.AddQuantize();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate memory for input and output tensors
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG_INF, "Failed to allocate tensors!");
        return false;
    }

    // Get pointers for input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;

    // Print input and output tensor types and dimensions
    ESP_LOGI(TAG_INF, "Input tensor type: %s, shape: %d, %d, %d, %d",
             TfLiteTypeGetName(input->type), input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG_INF, "Output tensor type: %s, shape: %d, %d, zero_point: %d, scale: %.6f",
             TfLiteTypeGetName(output->type), output->dims->data[0], output->dims->data[1], output->params.zero_point, output->params.scale);

    if (output->dims->data[1] != NUM_CLASSES)
    {
        ESP_LOGW(TAG_INF, "Warning: Model output size (%d) does not match NUM_CLASSES (%d)!", 
                 output->dims->data[1], NUM_CLASSES);
    }
    return true;
}

/**
 * @brief Set an input pixel directly into the interpreter's input tensor.
 *        Performs quantization: (val / 127.5 - 1.0) / scale + zero_point
 */
void inference_set_input_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= IMG_SIZE || y < 0 || y >= IMG_SIZE) return;

    const uint8_t channels[3] = {r, g, b};
    int base_idx = (y * IMG_SIZE + x) * 3;

    for (int c = 0; c < 3; ++c)
    {
        float val_float = ((float)channels[c] / 127.5f) - 1.0f;
        int32_t val_quant = static_cast<int32_t>(roundf(val_float / input_scale) + input_zero_point);
        
        if (val_quant > 127) val_quant = 127;
        if (val_quant < -128) val_quant = -128;
        
        input->data.int8[base_idx + c] = static_cast<int8_t>(val_quant);
    }
}

/**
 * @brief Run inference on the model, obtain the prediction and dequantize to float.
 * @param prediction Pointer to store the prediction result. Expected to be of size NUM_CLASSES.
 * @return True if inference was successful, false otherwise.
 */
bool inference_predict(float *prediction)
{
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        return false;
    }

    // Dequantize the output from int8 to float
    float sum = 0;
    for (size_t i = 0; i < NUM_CLASSES; ++i)
    {
        prediction[i] = (static_cast<float>(output->data.int8[i]) - output->params.zero_point) * output->params.scale;
        sum += prediction[i];
    }
    ESP_LOGD(TAG_INF, "Prediction sum: %.2f", sum);

    return true;
}
