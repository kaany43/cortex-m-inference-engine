#pragma once

#include "memory.h"
#include "uart.h"

void compute_peak_sram(int* layers, int num_layers) {
    volatile int peak_sram = 0;
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
}