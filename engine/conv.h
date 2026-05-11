#pragma once
#include <stdint.h>

void conv2d(const int8_t* input, int H, int W,
            const int8_t* kernel, int K,
            int8_t* output, int32_t scale_fixed);