#include <stdint.h>
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
int32_t dot_product_int8(int8_t* a, int8_t* b, int len);
static inline int32_t quantized(int32_t value, int32_t scale_fixed);
static inline int32_t relu(int32_t x);

// Helper function implementations
int32_t dot_product_int8(int8_t* a, int8_t* b, int len) {
    int32_t result = 0;
    for (int i = 0; i < len; i++) {
        result += (int32_t)a[i] * (int32_t)b[i];
    }
    return result;
}

static inline int32_t quantized(int32_t value, int32_t scale_fixed) {
    return ((int64_t)value * scale_fixed) >> 8;
}

static inline int32_t relu(int32_t x) {
    return (x > 0) ? x : 0;
}

void run_layer(int8_t* input, int8_t* weights, int8_t* output, int rows, int cols) {
    int32_t raw_dp[16];
    int32_t max_val = 0;

    // 1. Pass: Compute raw dot products and find the maximum positive value
    for (int r = 0; r < rows; r++) {
        raw_dp[r] = dot_product_int8(input, weights + (r * cols), cols);
        if (raw_dp[r] > max_val) {
            max_val = raw_dp[r];
        }
    }

    // Prevent division by zero if all values are <= 0
    if (max_val == 0) {
        max_val = 1;
    }

    // Calculate fixed-point scale factor (maps max_val to 127 using 8-bit shift)
    // This entirely avoids floating-point operations!
    int32_t scale_fixed = (127 << 8) / max_val;

    // 2. Pass: Quantize and apply ReLU
    for (int r = 0; r < rows; r++) {
        int32_t scaled = quantized(raw_dp[r], scale_fixed);
        output[r] = relu(scaled);

        print_char('O');
        print_char(':');
        print_int(output[r]);
        print_char('\n');
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
    print_char('O');
    print_char('K');
    print_char('\n');


    print_char('P');
    print_char(':');
    print_int(peak_sram);
    print_char('\n');




    int8_t input[] = {1, 2, 3};
    // Example weight matrices
    int8_t weights1[2][3] = {{1, 0, -1}, {2, 1, 0}};
    int8_t weights2[2][2] = {{1, -1}, {0, 1}};
    
    int8_t hidden_out[2];
    int8_t output[2];

    print_char('L'); print_char('1'); print_char('\n');
    run_layer(input, (int8_t*)weights1, hidden_out, 2, 3);

    print_char('L'); print_char('2'); print_char('\n');
    run_layer(hidden_out, (int8_t*)weights2, output, 2, 2);

    return 0;
}