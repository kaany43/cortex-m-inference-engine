extern int main();

void Reset_Handler() {
    main();
    while(1) {}
}

__attribute__((section(".vectors")))
void (*vectors[])() = {
    (void(*)())0x20008000,  // stack pointer
    Reset_Handler
};