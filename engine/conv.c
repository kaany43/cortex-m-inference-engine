#include "conv.h"

void conv2d(const int8_t* input, int H, int W, const int8_t* kernel, int K, int8_t* output, int32_t scale_fixed) {
    for (int h = 0; h <= H - K; h++) {
        for (int w = 0; w <= W - K; w++) {
            int32_t sum = 0;
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    sum += input[(h + kh) * W + (w + kw)] * kernel[kh * K + kw];
                }
            }
            int32_t scaled = quantized(sum, scale_fixed); 
            int32_t relu_val = relu(scaled);
            output[h * (W - K + 1) + w] = (int8_t)(relu_val > 127 ? 127 : relu_val); 
        }
    }
}