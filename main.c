#include <stdint.h>
#define UART0_DR (*((volatile unsigned int*)0x4000C000))

void print_char(char c) {
    UART0_DR = c;
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
    print_char('O');
    print_char('K');
    print_char('\n');

    void print_int(int n) {
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
    print_char('P');
    print_char(':');
    print_int(peak_sram);
    print_char('\n');


    int32_t dot_product_int8(int8_t* a, int8_t* b, int len) {
        int32_t result = 0;
        for (int i = 0; i < len; i++) {
            result += (int32_t)a[i] * (int32_t)b[i];
        }
        return result;
    }
    int8_t a[] = {1, 2, 3};
    int8_t b[] = {4, 5, 6};
    int32_t result = dot_product_int8(a, b, 3);
    print_char('D');
    print_char(':');
    print_int(result);
    print_char('\n');
    int32_t scale_fixed = 13;
    int32_t result_quantized = ((int64_t)result * scale_fixed) >> 8;
    print_char('Q');
    print_char(':');
    print_int(result_quantized);
    print_char('\n');
    while(1) {

    }
    return 0;
}