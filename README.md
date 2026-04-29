# Cortex-M Inference Engine

A lightweight neural network inference engine for ARM Cortex-M4, written in C without external dependencies.

## What is this?
A bare-metal inference engine that runs on ARM Cortex-M4 microcontrollers.

## Target Hardware
ARM Cortex-M4 (tested on QEMU lm3s6965evb emulator)

## Build
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -nostdlib -T link.ld -o main.elf startup.c main.c
