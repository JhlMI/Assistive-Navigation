// ============================================================================
//  harness_pico - Haptic harness controller (Raspberry Pi Pico)
//
//  Two haptic channels, felt on different places of the body:
//    shoulders : micro-navigation from the vision (sidewalk and obstacles)
//    chest     : macro-navigation from the GPS route (turns, hazards, arrival)
//
//  Shoulder commands (one letter per line, resent by the Jetson every 0.15 s):
//    F  forward            : silence (silence = keep walking)
//    L  correct left       : left shoulder pulses
//    R  correct right      : right shoulder pulses
//    X  obstacle, STOP     : fast strong pulses on both shoulders
//    W  no sidewalk / path : same pattern as X (stop, check with the cane); chest stays active
//    Z  end of block       : fast pulses on all four motors for 1.5 s, then silence (one shot);
//                            X still works, the chest resumes after the burst
//    E  system fault       : all four motors 1 s, then silence (one shot)
//    S  silence            : guidance stopped
//  Chest commands (route), resent every 0.15 s too:
//    N,0   off                       N,6   off course, path to the left
//    N,1   turn soon, left           N,7   off course, path to the right
//    N,2   turn soon, right          N,8   walking the wrong way (turn around)
//    N,3   go left (pulse like L)     N,9   arrived (one long pulse, once)
//    N,4   go right (pulse like R)    N,10  turn now left  + hazard right after
//    N,5   stairs / hazard ahead     N,11  turn now right + hazard right after
//    While X is active the chest stays silent (one message at a time). During W it keeps
//    working: losing the sidewalk is typical at corners, where the turn cue is needed.
//    The Jetson currently uses only N,0 N,3 N,4 N,9.
//  Test / setup commands:
//    P        test all 6 outputs
//    C,n,d    output n (1..6) at duty d (0.00..100.00 %), manual mode
//    V,min,max,ms  shoulder pulse floor, peak and period
//    K,d      chest duty (%)
//    T,ms     timeout (0 = disabled). Default 1000 ms. On timeout: E once, then silence
//
//  Sends every 200 ms:
//    B,v_ina,i_ina,p_ina,v_bat_adc,v_jet_adc,mode,chest
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
#define PIN_LEFT      3        // TB1 PWMA  left shoulder
#define PIN_RIGHT     2        // TB1 PWMB  right shoulder
#define PIN_CHEST_L   4        // TB2 PWMA  left chest  (output 3)
#define PIN_CHEST_R   9        // TB2 PWMB  right chest (output 4)
static const uint OUTPUTS[6] = { 2, 3, 4, 9, 11, 15 };   // TB1 A/B, TB2 A/B, TB3 A/B

#define PIN_SDA       0        // INA219
#define PIN_SCL       1
#define PIN_VJET      27       // ADC1: DROK output (100k/20k divider)
#define PIN_VBAT      28       // ADC2: battery (100k/20k divider)
#define PIN_LED       25

// ---- PWM -------------------------------------------------------------------
#define PWM_FREQ_HZ   20000
#define DUTY_MAX      70.00f   // 3 V motors with VM = 5 V (shoulders)
#define CHEST_MAX     80.00f   // chest: weaker contact; the triangular pulse averages half of it

// ---- Shoulder patterns -----------------------------------------------------
static float    pulse_min  = 30.00f;   // ERMs barely spin below ~30 %
static float    pulse_max  = 70.00f;
static uint32_t period_ms  = 600;      // rise + fall
static uint32_t timeout_ms = 1000;     // no commands -> E once, then silence
#define ALERT_MS      120              // X: on/off time
#define CAREFUL_MS    1000             // W: one soft pulse per second
#define CAREFUL_ON_MS 250
#define CAREFUL_DUTY  45.0f
#define FAULT_MS      1000             // E: all motors on for 1 s
#define CURB_MS       1500             // Z: fast pulses on all motors, then silence

// ---- Chest patterns --------------------------------------------------------
static float chest_duty = 80.0f;       // set with K,d after the intensity test
#define TAP_MS        120              // short tap
#define GAP_MS        120

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
void pwm_limit(uint pin, float duty, float limit) {
    if (duty < 0.0f)  duty = 0.0f;
    if (duty > limit) duty = limit;
    uint slice = pwm_gpio_to_slice_num(pin);
    uint32_t period = pwm_hw->slice[slice].top + 1;
    pwm_set_gpio_level(pin, (uint16_t)(duty * period / 100.0f));
}

void pwm(uint pin, float duty) { pwm_limit(pin, duty, DUTY_MAX); }

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

// X: on/off every ALERT_MS at full allowed duty
float alert(uint32_t t_ms) {
    return ((t_ms / ALERT_MS) % 2 == 0) ? DUTY_MAX : 0.0f;
}

// W: one soft pulse per second
float careful(uint32_t t_ms) {
    return ((t_ms % CAREFUL_MS) < CAREFUL_ON_MS) ? CAREFUL_DUTY : 0.0f;
}

// n short taps at the start of every cycle_ms
bool taps(uint32_t t_ms, int n, uint32_t cycle_ms) {
    uint32_t t = t_ms % cycle_ms;
    uint32_t slot = TAP_MS + GAP_MS;
    if (t >= (uint32_t)n * slot) return false;
    return (t % slot) < TAP_MS;
}

// Chest pattern -> duty for the left and right chest motors
void chest_pattern(int code, uint32_t t, float *l, float *r) {
    float d = chest_duty;
    bool on;
    *l = 0.0f; *r = 0.0f;
    switch (code) {
    case 1: if (taps(t, 2, 3000)) *l = d; break;                     // turn soon left
    case 2: if (taps(t, 2, 3000)) *r = d; break;                     // turn soon right
    case 3: *l = (pulse_max > 0.0f ? d * pulse(t) / pulse_max : 0.0f); break;                    // go left: same soft pulse as L
    case 4: *r = (pulse_max > 0.0f ? d * pulse(t) / pulse_max : 0.0f); break;                    // go right: same soft pulse as R
    case 5: if (taps(t, 3, 2000)) { *l = d; *r = d; } break;         // stairs / hazard
    case 6: if ((t % 2000) < 600) *l = d; break;                     // off course: path left
    case 7: if ((t % 2000) < 600) *r = d; break;                     // off course: path right
    case 8: {                                                        // turn around: L-R-L
        uint32_t k = (t % 2400) / 300;
        if (k == 0 || k == 4) *l = d;
        else if (k == 2) *r = d;
        break; }
    case 9: if (t < 1500) { *l = d; *r = d; } break;                 // arrived: once
    case 10: case 11: {                                              // turn now + hazard
        uint32_t c = t % 2400;
        if (c < 1200) on = (c % 400) < 250;                          // turn side 1.2 s
        else on = false;
        if (on) { if (code == 10) *l = d; else *r = d; }
        if (c >= 1400 && taps(c - 1400, 3, 1000)) { *l = d; *r = d; } // then the hazard taps
        break; }
    default: break;
    }
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
    printf("\n=== harness_pico v2.3 | PWM %d Hz | limit %.2f %% | chest %.0f %% | INA219 %s ===\n",
           PWM_FREQ_HZ, DUTY_MAX, chest_duty, ina_ok ? "OK" : "NOT RESPONDING");

    char mode = 'S';                       // F, L, R, X, W, Z, E, S or C (manual)
    int  chest = 0;                        // chest pattern code (0 = off)
    uint32_t now = to_ms_since_boot(get_absolute_time());
    uint32_t t0 = now, t_cmd = now, t_tel = now, t_led = now, t_chest = now, t_chest_cmd = now;
    bool led = false;

    while (true) {
        now = to_ms_since_boot(get_absolute_time());

        // ---- Commands -------------------------------------------------------
        if (read_line()) {
            char letter = (char)toupper((unsigned char)line[0]);
            int n;
            float d, d2;

            if (letter == 'F' || letter == 'L' || letter == 'R' || letter == 'X' ||
                letter == 'W' || letter == 'E' || letter == 'S' || letter == 'Z') {
                t_cmd = now;
                if (letter != mode)
                    t0 = now;              // every new pattern starts from the beginning
                if (mode == 'C') all_off();
                mode = letter;
            }
            else if (letter == 'N' && sscanf(line + 1, ",%d", &n) == 1 && n >= 0 && n <= 11) {
                t_chest_cmd = now;
                if (n != chest) t_chest = now;
                chest = n;
            }
            else if (letter == 'P') {
                mode = 'S';
                chest = 0;
                test_outputs();
            }
            else if (letter == 'C' && sscanf(line + 1, ",%d,%f", &n, &d) == 2 && n >= 1 && n <= 6) {
                if (mode != 'C') all_off();
                mode = 'C';
                chest = 0;
                pwm(OUTPUTS[n - 1], d);
                printf("Output %d -> %.2f %%\n", n, d);
            }
            else if (letter == 'V' && sscanf(line + 1, ",%f,%f,%d", &d2, &d, &n) == 3 && n >= 100 && d2 <= d) {
                pulse_min = d2;
                pulse_max = d;
                period_ms = (uint32_t)n;
                printf("Pulse: %.2f -> %.2f %%, period %d ms\n", d2, d, n);
            }
            else if (letter == 'K' && sscanf(line + 1, ",%f", &d) == 1 && d >= 0.0f) {
                chest_duty = (d > CHEST_MAX) ? CHEST_MAX : d;
                printf("Chest duty: %.2f %%\n", chest_duty);
            }
            else if (letter == 'T' && sscanf(line + 1, ",%d", &n) == 1 && n >= 0) {
                timeout_ms = (uint32_t)n;
                printf("Timeout: %d ms\n", n);
            }
            else {
                printf("E,invalid command: %s\n", line);
            }
        }

        // ---- Safety: no commands -> fault pattern once, then silence ----------
        if (timeout_ms > 0 && mode != 'S' && mode != 'C' && mode != 'E' && now - t_cmd > timeout_ms) {
            mode = 'E';
            t0 = now;
            chest = 0;
            printf("E,timeout\n");
        }
        if (timeout_ms > 0 && chest != 0 && now - t_chest_cmd > timeout_ms) chest = 0;

        // ---- Outputs ---------------------------------------------------------
        uint32_t t = now - t0;
        float sl = 0.0f, sr = 0.0f, cl = 0.0f, cr = 0.0f;
        switch (mode) {
        case 'L': sl = pulse(t); break;
        case 'R': sr = pulse(t); break;
        case 'X': sl = sr = alert(t); break;
        case 'W': sl = sr = alert(t); break;     // same feel as X, but the chest keeps guiding
        case 'E': if (t < FAULT_MS) sl = sr = cl = cr = DUTY_MAX; break;
        case 'Z': if (t < CURB_MS) sl = sr = cl = cr = alert(t); break;
        case 'F': case 'S': default: break;      // silence
        }
        if (mode == 'C') {
            // manual: outputs keep the values set by C,n,d
        } else {
            if (mode != 'X' && mode != 'E' && mode != 'S' && !(mode == 'Z' && t < CURB_MS))
                chest_pattern(chest, now - t_chest, &cl, &cr);   // chest silent during X, E and S
            pwm(PIN_LEFT, sl);
            pwm(PIN_RIGHT, sr);
            pwm_limit(PIN_CHEST_L, cl, CHEST_MAX);
            pwm_limit(PIN_CHEST_R, cr, CHEST_MAX);
        }

        // ---- Telemetry -------------------------------------------------------
        if (now - t_tel >= TELEMETRY_MS) {
            t_tel = now;
            float v = -1.0f, i = -1.0f, pw = -1.0f;
            if (!ina_ok) ina_ok = ina_configure();          // retry if plugged in later
            if (ina_ok && !ina_measure(&v, &i, &pw)) { v = -1.0f; i = -1.0f; pw = -1.0f; ina_ok = false; }
            float vbat = adc_voltage(2, VBAT_FACTOR);       // GP28
            float vjet = adc_voltage(1, VJET_FACTOR);       // GP27
            printf("B,%.3f,%.4f,%.3f,%.3f,%.3f,%c,%d\n", v, i, pw, vbat, vjet, mode, chest);
        }

        // ---- LED: slow when silent, fast while a pattern is active -----------
        bool active = (mode != 'S' && mode != 'F') || chest != 0;
        if (now - t_led >= (active ? 100u : 500u)) {
            t_led = now;
            led = !led;
            gpio_put(PIN_LED, led);
        }

        sleep_ms(5);
    }
}