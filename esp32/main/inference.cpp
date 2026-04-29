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
static int8_t quantization_lut[256];

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

    // Load TFlite model from the C array
    model = tflite::GetModel(MODEL_TFLITE);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG_INF, "Model schema mismatch!");
        return false;
    }

    // Create an interpreter using MicroMutableOpResolver
    static tflite::MicroMutableOpResolver<15> resolver;
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
    resolver.AddPad(); 
    resolver.AddMul();

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

    if (input_scale == 0) {
        ESP_LOGW(TAG_INF, "Input scale is 0, defaulting to 1/255.0");
        input_scale = 1.0f / 255.0f;
    }

    // Initialize Quantization LUT
    for (int i = 0; i < 256; i++) {
        // Map [0, 255] to [-1.0, 1.0] (tf mode)
        float val_float = ((float)i / 127.5f) - 1.0f;
        int32_t val_quant = static_cast<int32_t>(roundf(val_float / input_scale) + input_zero_point);
        if (val_quant > 127) val_quant = 127;
        if (val_quant < -128) val_quant = -128;
        quantization_lut[i] = static_cast<int8_t>(val_quant);
    }

    // Print input and output tensor types and dimensions
    ESP_LOGI(TAG_INF, "Model loaded successfully");
    ESP_LOGI(TAG_INF, "Input: %s, shape: (%d, %d, %d, %d), scale: %.6f, zp: %d",
             TfLiteTypeGetName(input->type), input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3],
             input->params.scale, input->params.zero_point);
    ESP_LOGI(TAG_INF, "Output: %s, shape: (%d, %d), scale: %.6f, zp: %d",
             TfLiteTypeGetName(output->type), output->dims->data[0], output->dims->data[1], 
             output->params.scale, output->params.zero_point);

    if (output->dims->data[1] != NUM_CLASSES)
    {
        ESP_LOGW(TAG_INF, "Model output size (%d) does not match NUM_CLASSES (%d)!", 
                 output->dims->data[1], NUM_CLASSES);
    }
    
    return true;
}

/**
 * @brief Set an input pixel directly into the interpreter's input tensor.
 *        Uses a pre-calculated LUT for quantization.
 */
void inference_set_input_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= IMG_SIZE || y < 0 || y >= IMG_SIZE) return;

    int base_idx = (y * IMG_SIZE + x) * 3;
    input->data.int8[base_idx]     = quantization_lut[r];
    input->data.int8[base_idx + 1] = quantization_lut[g];
    input->data.int8[base_idx + 2] = quantization_lut[b];
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
        ESP_LOGE(TAG_INF, "Failed to invoke interpreter!");
        return false;
    }

    // Dequantize the output from int8 to float
    for (size_t i = 0; i < NUM_CLASSES; ++i)
    {
        prediction[i] = (static_cast<float>(output->data.int8[i]) - output->params.zero_point) * output->params.scale;
    }

    return true;
}
