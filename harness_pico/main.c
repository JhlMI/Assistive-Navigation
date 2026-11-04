// ============================================================================
//  harness_pico - Haptic harness controller (Raspberry Pi Pico)
//
//  Receives one letter per line from the Jetson (USB serial):
//    F        forward  : left and right pulse together
//    L        left     : only the left motor pulses
//    R        right    : only the right motor pulses
//    X        alert    : short, strong pulses on both (stop)
//    S        silence  : everything off
//  Test commands:
//    P        test all 6 outputs
//    C,n,d    output n (1..6) at duty d (0.00..100.00 %), manual mode
//    V,d,ms   navigation pulse max duty and period
//    T,ms     timeout (0 = disabled). Default 1000 ms
//
//  Sends every 200 ms:
//    B,v_ina,i_ina,p_ina,v_bat_adc,v_jet_adc,mode
//    (v_ina/i_ina/p_ina = -1 if the INA219 does not respond)
// ============================================================================
#include <stdio.h>
#include <ctype.h>
#include "pico/stdlib.h"
#include "hardware/pwm.h"
#include "hardware/clocks.h"
#include "hardware/i2c.h"
#include "hardware/adc.h"

// ---- Pins ------------------------------------------------------------------
#define PIN_STBY      14
#define PIN_LEFT      3        // TB1 PWMA
#define PIN_RIGHT     2        // TB1 PWMB
static const uint OUTPUTS[6] = { 2, 3, 4, 9, 11, 15 };   // TB1 A/B, TB2 A/B, TB3 A/B

#define PIN_SDA       0        // INA219
#define PIN_SCL       1
#define PIN_VJET      27       // ADC1: DROK output (100k/20k divider)
#define PIN_VBAT      28       // ADC2: battery (100k/20k divider)
#define PIN_LED       25

// ---- PWM -------------------------------------------------------------------
#define PWM_FREQ_HZ   20000
#define DUTY_MAX      70.00f   // 3 V motors with VM = 5 V

// ---- Navigation ------------------------------------------------------------
static float    pulse_min  = 30.00f;   // navigation pulse floor (%): ERMs barely spin below ~30 %
static float    pulse_max  = 70.00f;   // navigation pulse peak (%)
static uint32_t period_ms  = 600;      // rise + fall
static uint32_t timeout_ms = 1000;     // no commands -> silence
#define ALERT_MS      120              // alert on/off time

// ---- Measurements ----------------------------------------------------------
#define VBAT_FACTOR   6.001f           // calibrated against multimeter: 16.20 V
#define VJET_FACTOR   6.017f           // calibrated against multimeter: 12.27 V
#define TELEMETRY_MS  200

#define INA_ADDR      0x40
#define INA_R_SHUNT   0.1f             // stock R100
#define INA_I_LSB     0.0001f          // 0.1 mA per bit -> ~3.2 A range

// ============================================================================
//  PWM
// ============================================================================
void pwm_init_pin(uint pin) {
    uint slice = pwm_gpio_to_slice_num(pin);
    gpio_set_function(pin, GPIO_FUNC_PWM);
    pwm_set_clkdiv(slice, 1.0f);
    pwm_set_wrap(slice, clock_get_hz(clk_sys) / PWM_FREQ_HZ - 1);
    pwm_set_gpio_level(pin, 0);
    pwm_set_enabled(slice, true);
}

// Duty cycle in % (0.00 .. 100.00), limited to DUTY_MAX
void pwm(uint pin, float duty) {
    if (duty < 0.0f)     duty = 0.0f;
    if (duty > DUTY_MAX) duty = DUTY_MAX;
    uint slice = pwm_gpio_to_slice_num(pin);
    uint32_t period = pwm_hw->slice[slice].top + 1;
    pwm_set_gpio_level(pin, (uint16_t)(duty * period / 100.0f));
}

void all_off(void) {
    for (int i = 0; i < 6; i++) pwm(OUTPUTS[i], 0.0f);
}

void test_outputs(void) {
    all_off();
    for (int i = 0; i < 6; i++) {
        printf("Output %d (GP%u)\n", i + 1, OUTPUTS[i]);
        pwm(OUTPUTS[i], DUTY_MAX);
        sleep_ms(1000);
        pwm(OUTPUTS[i], 0.0f);
        sleep_ms(300);
    }
    printf("Test finished\n");
}

// Triangle pulse: 0 -> pulse_max -> 0 over period_ms
float pulse(uint32_t t_ms) {
    float phase = (float)(t_ms % period_ms) / (float)period_ms;
    float tri = (phase < 0.5f) ? 2.0f * phase : 2.0f * (1.0f - phase);
    return pulse_max * tri;
}

// Alert: on/off every ALERT_MS at full allowed duty
float alert(uint32_t t_ms) {
    return ((t_ms / ALERT_MS) % 2 == 0) ? DUTY_MAX : 0.0f;
}

// ============================================================================
//  INA219
// ============================================================================
static bool ina_ok = false;

bool ina_write(uint8_t reg, uint16_t value) {
    uint8_t buf[3] = { reg, (uint8_t)(value >> 8), (uint8_t)value };
    return i2c_write_timeout_us(i2c0, INA_ADDR, buf, 3, false, 2000) == 3;
}

bool ina_read(uint8_t reg, uint16_t *value) {
    uint8_t buf[2];
    if (i2c_write_timeout_us(i2c0, INA_ADDR, &reg, 1, true, 2000) != 1) return false;
    if (i2c_read_timeout_us(i2c0, INA_ADDR, buf, 2, false, 2000) != 2) return false;
    *value = ((uint16_t)buf[0] << 8) | buf[1];
    return true;
}

uint16_t ina_calibration(void) {
    return (uint16_t)(0.04096f / (INA_I_LSB * INA_R_SHUNT));   // 4096
}

bool ina_configure(void) {
    // 32 V bus, PGA +-320 mV, 12 bits averaged x16, continuous mode
    return ina_write(0x00, 0x3E67) && ina_write(0x05, ina_calibration());
}

void ina_init(void) {
    i2c_init(i2c0, 400000);
    gpio_set_function(PIN_SDA, GPIO_FUNC_I2C);
    gpio_set_function(PIN_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(PIN_SDA);
    gpio_pull_up(PIN_SCL);
    ina_ok = ina_configure();
}

// Returns false if the INA219 does not respond
bool ina_measure(float *v, float *i, float *p) {
    uint16_t bus, cur, pow;
    if (!ina_write(0x05, ina_calibration())) return false;   // in case it reset
    if (!ina_read(0x02, &bus) || !ina_read(0x04, &cur) || !ina_read(0x03, &pow)) return false;
    *v = (float)(bus >> 3) * 0.004f;
    *i = -(float)(int16_t)cur * INA_I_LSB;   // VIN+/VIN- wired reversed
    *p = (float)pow * 20.0f * INA_I_LSB;
    return true;
}

// ============================================================================
//  ADC (average of 16 readings)
// ============================================================================
float adc_voltage(uint input, float factor) {
    adc_select_input(input);
    uint32_t sum = 0;
    for (int k = 0; k < 16; k++) sum += adc_read();
    return (sum / 16.0f) * 3.3f / 4095.0f * factor;
}

// ============================================================================
//  Non-blocking line reader
// ============================================================================
static char line[32];
static int  length = 0;

bool read_line(void) {
    int c;
    while ((c = getchar_timeout_us(0)) != PICO_ERROR_TIMEOUT) {
        if (c == '\r' || c == '\n') {
            if (length == 0) continue;
            line[length] = '\0';
            length = 0;
            return true;
        }
        if (length < (int)sizeof(line) - 1) line[length++] = (char)c;
        else length = 0;
    }
    return false;
}

// ============================================================================
int main(void) {
    stdio_init_all();

    gpio_init(PIN_LED);
    gpio_set_dir(PIN_LED, GPIO_OUT);

    gpio_init(PIN_STBY);
    gpio_set_dir(PIN_STBY, GPIO_OUT);
    gpio_put(PIN_STBY, 0);
    for (int i = 0; i < 6; i++) pwm_init_pin(OUTPUTS[i]);
    gpio_put(PIN_STBY, 1);

    adc_init();
    adc_gpio_init(PIN_VJET);
    adc_gpio_init(PIN_VBAT);
    ina_init();

    sleep_ms(1500);
    printf("\n=== harness_pico | PWM %d Hz | limit %.2f %% | INA219 %s ===\n",
           PWM_FREQ_HZ, DUTY_MAX, ina_ok ? "OK" : "NOT RESPONDING");

    char mode = 'S';                       // F, L, R, X, S or C (manual)
    uint32_t now = to_ms_since_boot(get_absolute_time());
    uint32_t t0 = now, t_cmd = now, t_tel = now, t_led = now;
    bool led = false;

    while (true) {
        now = to_ms_since_boot(get_absolute_time());

        // ---- Commands -------------------------------------------------------
        if (read_line()) {
            char letter = (char)toupper((unsigned char)line[0]);
            int n;
            float d,d2;
            t_cmd = now;

            if (letter == 'F' || letter == 'L' || letter == 'R' || letter == 'X' || letter == 'S') {
                if (letter != mode && (mode == 'S' || mode == 'C' || letter == 'X' || mode == 'X'))
                    t0 = now;              // restart the pattern from the beginning
                mode = letter;
            }
            else if (letter == 'P') {
                mode = 'S';
                test_outputs();
            }
            else if (letter == 'C' && sscanf(line + 1, ",%d,%f", &n, &d) == 2 && n >= 1 && n <= 6) {
                if (mode != 'C') all_off();
                mode = 'C';
                pwm(OUTPUTS[n - 1], d);
                printf("Output %d -> %.2f %%\n", n, d);
            }
            else if (letter == 'V' && sscanf(line + 1, ",%f,%f,%d", &d2, &d, &n) == 3 && n >= 100 && d2 <= d) {
                pulse_min = d2;
                pulse_max = d;
                period_ms = (uint32_t)n;
                printf("Pulse: %.2f -> %.2f %%, period %d ms\n", d2, d, n);
            }
            else if (letter == 'T' && sscanf(line + 1, ",%d", &n) == 1 && n >= 0) {
                timeout_ms = (uint32_t)n;
                printf("Timeout: %d ms\n", n);
            }
            else {
                printf("E,invalid command: %s\n", line);
            }
        }

        // ---- Safety: no commands -> silence ----------------------------------
        if (timeout_ms > 0 && mode != 'S' && mode != 'C' && now - t_cmd > timeout_ms) {
            mode = 'S';
            printf("E,timeout\n");
        }

        // ---- Outputs ---------------------------------------------------------
        float p = pulse(now - t0);
        switch (mode) {
        case 'F': pwm(PIN_LEFT, p);    pwm(PIN_RIGHT, p);    break;
        case 'L': pwm(PIN_LEFT, p);    pwm(PIN_RIGHT, 0.0f); break;
        case 'R': pwm(PIN_LEFT, 0.0f); pwm(PIN_RIGHT, p);    break;
        case 'X': pwm(PIN_LEFT, alert(now - t0)); pwm(PIN_RIGHT, alert(now - t0)); break;
        case 'S': all_off();                                 break;
        default:  break;                                     // 'C': manual
        }

        // ---- Telemetry -------------------------------------------------------
        if (now - t_tel >= TELEMETRY_MS) {
            t_tel = now;
            float v = -1.0f, i = -1.0f, pw = -1.0f;
            if (!ina_ok) ina_ok = ina_configure();          // retry if plugged in later
            if (ina_ok && !ina_measure(&v, &i, &pw)) { v = -1.0f; i = -1.0f; pw = -1.0f; ina_ok = false; }
            float vbat = adc_voltage(2, VBAT_FACTOR);       // GP28
            float vjet = adc_voltage(1, VJET_FACTOR);       // GP27
            printf("B,%.3f,%.4f,%.3f,%.3f,%.3f,%c\n", v, i, pw, vbat, vjet, mode);
        }

        // ---- LED: slow when silent, fast while a pattern is active -----------
        if (now - t_led >= ((mode == 'S') ? 500u : 100u)) {
            t_led = now;
            led = !led;
            gpio_put(PIN_LED, led);
        }

        sleep_ms(5);
    }
}
