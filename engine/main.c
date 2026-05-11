#include <stdint.h>

#define UART0_DR (*((volatile unsigned int*)0x4000C000))

#include "../model/headers/model.h"
#include "uart.c"
#include "inference.c"
#include "conv.c"
#include "memory.c"

int main() {
    int layers[] = {784, 32, 10};
    int num_layers = 3;

    compute_peak_sram(layers, num_layers);
    int correct_predictions = 0;

    for (int8_t i = 0; i < TEST_COUNT; i++) {

        int8_t hidden_out[LAYER1_ROWS];
        int8_t output[LAYER2_ROWS];
        int8_t conv_out[676]; // 26x26 output from 28x28 input and 3x3 kernel

        conv2d(test_batch + (i * 784), 28, 28, conv1_weights, 3, conv_out, CONV1_SCALE_FIXED);
        run_layer(conv_out, layer1_weights, layer1_bias, hidden_out, LAYER1_ROWS, LAYER1_COLS, LAYER1_SCALE_FIXED);
        final_layer(hidden_out, layer2_weights, layer2_bias, output, LAYER2_ROWS, LAYER2_COLS, i, &correct_predictions, LAYER2_SCALE_FIXED);

    }

    print_char('C');
    print_char(':');
    print_int(correct_predictions);
    print_char('\n');

    return 0;
}