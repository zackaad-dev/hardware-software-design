#pragma once

#include <stdint.h>

bool inference_init();
void inference_set_input_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);
bool inference_predict(float *prediction);
