cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(harness_pico C CXX ASM)
set(CMAKE_C_STANDARD 11)

pico_sdk_init()

add_executable(harness_pico main.c)
target_link_libraries(harness_pico pico_stdlib hardware_pwm hardware_clocks hardware_i2c hardware_adc)

pico_enable_stdio_usb(harness_pico 1)
pico_enable_stdio_uart(harness_pico 0)
pico_add_extra_outputs(harness_pico)
