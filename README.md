# Cortex-M Inference Engine

A lightweight neural network inference engine for ARM Cortex-M4, written in C without external dependencies.

## What is this?
A bare-metal inference engine that runs on ARM Cortex-M4 microcontrollers.
No OS, no libraries, no runtime — just hardware and math.

## Current Status
- [x] Bare-metal ARM setup (vector table, linker script)
- [x] UART output without printf
- [x] Static SRAM memory analyzer
- [x] Dot product (core neural network operation)
- [x] INT8 quantized layer execution
- [ ] Layer-by-layer memory scheduler
- [ ] Benchmark vs TFLite Micro

## Target Hardware
ARM Cortex-M4 (tested on QEMU lm3s6965evb emulator)

## Build
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -nostdlib -T link.ld -o main.elf startup.c main.c