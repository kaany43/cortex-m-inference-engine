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




    // Example input vector
    int8_t input[] = {1, 2, 3};
    // Example weight matrix (3 rows, 3 columns)
    int8_t weights[3][3] = {
        {1, 0, -1},
        {2, 1, 0},
        {0, -2, 2}
    };
    const int rows = 3;
    const int cols = 3;
    int32_t scale_fixed = 13;
    for (int r = 0; r < rows; r++) {
        int32_t dp = dot_product_int8(input, weights[r], cols);
        int32_t scaled = quantized(dp, scale_fixed);
        int32_t activated = relu(scaled);
        // Print raw dot product
        print_char('D');
        print_char(':');
        print_int(dp);
        print_char('\n');
        // Print scaled value
        print_char('S');
        print_char(':');
        print_int(scaled);
        print_char('\n');
        // Print activated (ReLU) value
        print_char('R');
        print_char(':');
        print_int(activated);
        print_char('\n');
    }

    while(1) {

    }
    return 0;
}