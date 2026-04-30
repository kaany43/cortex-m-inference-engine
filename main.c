#include <stdint.h>
#include "layer1_weights.h"
#include "layer2_weights.h"
#include "test_image.h"
#include "test_batch.h"
#include "scales.h"
#include "bias.h"
#define UART0_DR (*((volatile unsigned int*)0x4000C000))
void print_char(char c);

void print_int(int32_t n) {
    if (n < 0) {
        print_char('-');
        n = -n;
    }
    if (n == 0) {
        print_char('0');
        return;
    }
    char buf[10];
    int i = 0;
    while (n > 0) {
        buf[i++] = '0' + (n % 10);
        n /= 10;
    }
    for (int j = i - 1; j >= 0; j--) {
        print_char(buf[j]);
    }
}

void print_char(char c) {
    UART0_DR = c;
}

// Helper function prototypes
int32_t dot_product_int8(const int8_t* a, const int8_t* b, int len);
static inline int32_t quantized(int32_t value, int32_t scale_fixed);
static inline int32_t relu(int32_t x);

// Helper function implementations
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
    //Print and compare with actual label
    print_char('A');
    print_char(':');
    print_int(argmax);
    print_char('\n');
    print_char('L');
    print_char(':');
    print_int(test_labels[i]);
    print_char('\n');

    // Count correct predictions
    if (argmax == test_labels[i]) {
        (*correct_predictions)++;
    }
}

int main() {
    volatile int peak_sram = 0;
    int layers[] = {256, 512, 128, 64};
    int num_layers = 4;

    for (int i = 0; i < num_layers - 1; i++) {
        int used = layers[i] + layers[i + 1];
        if (used > peak_sram) {
            peak_sram = used;
        }
    }


    print_char('P');
    print_char(':');
    print_int(peak_sram);
    print_char('\n');




    int correct_predictions = 0;

    for (int8_t i = 0; i < TEST_COUNT; i++) {

        int8_t hidden_out[32];
        int8_t output[10];

        run_layer(test_batch + (i * 784), layer1_weights, layer1_bias, hidden_out, 32, 784, LAYER1_SCALE_FIXED);
        final_layer(hidden_out, layer2_weights, layer2_bias, output, 10, 32, i, &correct_predictions, LAYER2_SCALE_FIXED);

    }
    
    print_char('T');
    print_char('o');
    print_char('t');
    print_char('a');
    print_char('l');
    print_char(' ');
    print_char('C');
    print_char(':');
    print_int(correct_predictions);
    print_char('\n');

    return 0;
}