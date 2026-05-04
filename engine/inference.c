#include "uart.h"

int32_t dot_product_int8(const int8_t* a, const int8_t* b, int len) {
    int32_t result = 0;
    for (int i = 0; i < len; i++) {
        result += (int32_t)a[i] * (int32_t)b[i];
    }
    return result;
}

static inline int32_t quantized(int32_t value, int32_t scale_fixed) {
    return (int32_t)(((int64_t)value * scale_fixed) >> 16);
}

static inline int32_t relu(int32_t x) {
    return (x > 0) ? x : 0;
}

//Hidden layers
void run_layer(const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, int rows, int cols, int32_t scale_fixed) {
    for (int r = 0; r < rows; r++) {
        int32_t dp = dot_product_int8(input, weights + (r * cols), cols);
        int32_t scaled = quantized(dp, scale_fixed) + bias[r];
        int32_t relu_val = relu(scaled);
        output[r] = (int8_t)(relu_val > 127 ? 127 : relu_val);
    }
}
//Output layer
void final_layer(const int8_t* input, const int8_t* weights, const int32_t* bias, int8_t* output, int rows, int cols, int8_t i, int* correct_predictions, int32_t scale_fixed) {
    int32_t max_val = -2147483648;
    int32_t argmax = 0;

    for (int r = 0; r < rows; r++) {
        int32_t dp = dot_product_int8(input, weights + (r * cols), cols);
        int32_t scaled = quantized(dp, scale_fixed) + bias[r];

        if (scaled > max_val) {
            max_val = scaled;
            argmax = r;
        }

        int32_t relu_val = relu(scaled);
        output[r] = (int8_t)(relu_val > 127 ? 127 : relu_val);
    }
    // Count correct predictions
    if (argmax == test_labels[i]) {
        (*correct_predictions)++;
    }
}