#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_attr.h"
#include "esp_camera.h"

// Project includes
#include "model.h"
#include "inference.h"

// LED pin
#define LED_PIN GPIO_NUM_21

static float prediction[NUM_CLASSES];
static const char *TAG_MAIN = "Main";

// Standard ESP32-S3 camera pins (e.g. Freenove ESP32-S3 WROOM CAM / ESP32-S3 Sense)
// You may need to adapt these pins if you use a different board.
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sccb_sda = SIOD_GPIO_NUM,
    .pin_sccb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,

    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_RGB565,
    .frame_size = FRAMESIZE_240X240,

    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    .sccb_i2c_port = -1,
};

void setup(void)
{
    // Initialize inference
    if (!inference_init())
    {
        ESP_LOGE(TAG_MAIN, "Failed to initialize inference!");
        abort();
    }

    // Initialize camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK)
    {
        ESP_LOGE(TAG_MAIN, "Camera Init Failed");
        abort();
    }

    // Initialize LED
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED_PIN, 1); // Turn off LED (active low)
}

void loop(void)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
        ESP_LOGE(TAG_MAIN, "Camera capture failed");
        return;
    }

    // Convert RGB565 (240x240) to RGB888 and crop to IMG_SIZE x IMG_SIZE (224x224)
    // Directly push to inference input to save RAM
    int offset_x = (fb->width - IMG_SIZE) / 2;
    int offset_y = (fb->height - IMG_SIZE) / 2;
    
    // Ensure we don't go out of bounds if fb is smaller than expected
    if (offset_x < 0) offset_x = 0;
    if (offset_y < 0) offset_y = 0;

    uint8_t *pixels = fb->buf;
    for (int y = 0; y < IMG_SIZE; y++)
    {
        for (int x = 0; x < IMG_SIZE; x++)
        {
            int src_x = offset_x + x;
            int src_y = offset_y + y;
            
            // Boundary safety
            if (src_x >= fb->width || src_y >= fb->height) {
                continue;
            }
            
            // RGB565 (Big Endian) to RGB888 conversion
            // p[0] is RRRRRGGG, p[1] is GGGBBBBB
            uint8_t *p = &pixels[(src_y * fb->width + src_x) * 2];
            uint8_t r = p[0] & 0xF8;
            uint8_t g = ((p[0] & 0x07) << 5) | ((p[1] & 0xE0) >> 3);
            uint8_t b = (p[1] & 0x1F) << 3;

            inference_set_input_pixel(x, y, r, g, b);
        }
    }

    esp_camera_fb_return(fb);

    // Run inference
    if (!inference_predict(prediction))
    {
        ESP_LOGE(TAG_MAIN, "Failed to invoke interpreter!");
    }
    else
    {
        // Print output
        ESP_LOGI(TAG_MAIN, "Predictions:");
        for (int i = 0; i < NUM_CLASSES; i++) {
            ESP_LOGI(TAG_MAIN, "  Class %d: %.2f", i, prediction[i]);
        }

        // Assuming the class "johannes" was sorted first alphabetically, its index is 0.
        // If it was sorted differently, update JOHANNES_CLASS_INDEX.
        #define JOHANNES_CLASS_INDEX 0
        if (NUM_CLASSES > JOHANNES_CLASS_INDEX && prediction[JOHANNES_CLASS_INDEX] > 0.7f)
        {
            gpio_set_level(LED_PIN, 0); // Turn on LED (assuming active low)
        }
        else
        {
            gpio_set_level(LED_PIN, 1); // Turn off LED
        }
    }

    // Yield to allow other tasks (like IDLE and Watchdog) to run
    vTaskDelay(10 / portTICK_PERIOD_MS);
}

extern "C" void app_main(void)
{
    // Force output even if we crash immediately after
    printf("\n\n--- XIAO ESP32-S3 STARTING ---\n");
    printf("--- PSRAM ENABLED ---\n");
    fflush(stdout);
    usleep(200000); // 200ms delay to let serial settle

    ESP_LOGI(TAG_MAIN, "Starting app_main initialization...");
    setup();
    ESP_LOGI(TAG_MAIN, "Initialization complete, entering loop.");
    
    while (true)
    {
        loop();
    }
}
