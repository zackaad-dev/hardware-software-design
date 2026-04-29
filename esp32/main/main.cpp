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

// LED pin on XIAO ESP32S3 Sense (Active Low)
#define LED_PIN GPIO_NUM_21

static float prediction[NUM_CLASSES];
static const char *TAG_MAIN = "Main";

// Camera pins for XIAO ESP32S3 Sense
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
    .pin_d2 = Y4_GPIO_NUM, // Fixed: was Y2_GPIO_NUM
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
    gpio_set_level(LED_PIN, 1); // Start with LED OFF (Active Low)
}

static int reset = 0;

void loop(void)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
        ESP_LOGE(TAG_MAIN, "Camera capture failed");
        return;
    }

    // Crop and convert to RGB888 directly into inference input
    int offset_x = (fb->width - IMG_SIZE) / 2;
    int offset_y = (fb->height - IMG_SIZE) / 2;
    if (offset_x < 0) offset_x = 0;
    if (offset_y < 0) offset_y = 0;

    uint8_t *pixels = fb->buf;
    for (int y = 0; y < IMG_SIZE; y++)
    {
        for (int x = 0; x < IMG_SIZE; x++)
        {
            int src_x = offset_x + x;
            int src_y = offset_y + y;
            
            if (src_x >= fb->width || src_y >= fb->height) continue;
            
            // RGB565 (Big Endian: hb=RRRRRGGG, lb=GGGBBBBB)
            uint8_t hb = pixels[(src_y * fb->width + src_x) * 2];
            uint8_t lb = pixels[(src_y * fb->width + src_x) * 2 + 1];
            
            // Extract bits
            uint8_t r_raw = (hb & 0xF8) >> 3;
            uint8_t g_raw = ((hb & 0x07) << 3) | ((lb & 0xE0) >> 5);
            uint8_t b_raw = lb & 0x1F;

            // Bit-extension to 8-bit [0, 255]
            uint8_t r = (r_raw * 255) / 31;
            uint8_t g = (g_raw * 255) / 63;
            uint8_t b = (b_raw * 255) / 31;

            inference_set_input_pixel(x, y, r, g, b);
        }
    }
    esp_camera_fb_return(fb);

    // Run inference

    vTaskDelay(1);
    if (inference_predict(prediction))
    {
        const char* labels[] = {"member 1", "member 2", "nonmember"};
        
        // Find best prediction
        int max_idx = 0;
        float max_val = -1.0f;
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (prediction[i] > max_val) {
                max_val = prediction[i];
                max_idx = i;
            }
        }

        // Detailed log to understand distribution
        printf("Inf: %s (%.1f%%) | m1:%.2f m2:%.2f nm:%.2f\n", 
               labels[max_idx], max_val * 100.0f, prediction[0], prediction[1], prediction[2]);

        float threshold = 0.75f;
        if (max_val > threshold) {
            if (max_idx == 0 || max_idx == 1) { // member 1 or member 2
                ESP_LOGI(TAG_MAIN, ">>> ACCESS GRANTED: %s <<<", labels[max_idx]);
                gpio_set_level(LED_PIN, 0); // Active Low
				reset = 0;
            } else { // nonmember
                ESP_LOGI(TAG_MAIN, ">>> ACCESS DENIED: %s <<<", labels[max_idx]);
				if (reset == 0) {
					reset = 1;
					gpio_set_level(LED_PIN, 1); // Active Low
				}
            } 
		} else {
            // Low confidence detection
            if (reset == 0) {
                ESP_LOGI(TAG_MAIN, ">>> NO CONFIDENT DETECTION - LED OFF <<<");
                gpio_set_level(LED_PIN, 1); // Active Low
                reset = 1;
            }
        }
    }
}

extern "C" void app_main(void)
{
    printf("\n\n--- XIAO ESP32-S3 SENSE FACE RECOGNITION STARTING ---\n");
    fflush(stdout);
    vTaskDelay(200 / portTICK_PERIOD_MS);

    setup();
    
    while (true)
    {
        loop();
    }
}
