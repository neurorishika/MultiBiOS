 /**
 * @file v1_global_load.ino
 * @brief MultiBiOS Teensy 4.1 firmware: single GLOBAL_LOAD_REQ, independent RCK latches, 48-bit chain.
 * @version 1.0
 *
 * Wiring summary (example; match to your rig):
 *   SPI0: MOSI=11, SCK=13  (OE\ tied LOW, SRCLR\ tied HIGH on all TPIC6B595)
 *   GLOBAL_LOAD_REQ  -> Teensy pin 22  (change below if needed)
 *   RCK sense (from DAQ): Left=A=2, Right=B=3, SwitchL=C=4, SwitchR=D=5
 *   READY out to DAQ: OLFA_L=6, OLFA_R=7, SW_L=8, SW_R=9
 *   S-lines from DAQ:
 *     Left  olfactometer: S0=14, S1=15, S2=16
 *     Right olfactometer: S0=17, S1=18, S2=19
 *     Left/Right switch:  S =20 / 21
 *
 * Logical model:
 *   - On rising GLOBAL_LOAD_REQ: read all S-lines, build a single 48-bit frame:
 *       [OLFA_LEFT 16b] + [OLFA_RIGHT 16b] + [SW_LEFT 8b] + [SW_RIGHT 8b]
 *     shift it out (MSB-first). Then assert ALL READY lines HIGH.
 *   - An RCK rising edge for a given assembly commits its portion; we drop that assembly's READY.
 *   - A subsequent GLOBAL_LOAD_REQ rebuilds the frame (overwriting the preload) and re-asserts ALL READY.
 *
 * Timing:
 *   DAQ must provide preload_lead large enough for SPI shift time + margin.
 *   Example: 48 bits @ 10 MHz = 4.8 µs; 1 ms lead is extremely safe.
 */

#include <Arduino.h>
#include <SPI.h>

#ifndef BIT
#define BIT(n) (1u << (n))
#endif

// -------------------- Pins --------------------
constexpr int PIN_MOSI = 11;
constexpr int PIN_SCK  = 13;

// RCK sense inputs (from DAQ)
constexpr int PIN_RCK_SENSE_OLFA_L = 2;
constexpr int PIN_RCK_SENSE_SW_L   = 3;
constexpr int PIN_RCK_SENSE_OLFA_R = 4;
constexpr int PIN_RCK_SENSE_SW_R   = 5;

// READY outputs (to DAQ)
constexpr int PIN_READY_OLFA_L = 6;
constexpr int PIN_READY_SW_L   = 7;
constexpr int PIN_READY_OLFA_R = 8;
constexpr int PIN_READY_SW_R   = 9;

// S-line inputs (from DAQ)
constexpr int PIN_OLFA_L_S0 = 24, PIN_OLFA_L_S1 = 25, PIN_OLFA_L_S2 = 26;
constexpr int PIN_SW_L_S    = 27;
constexpr int PIN_OLFA_R_S0 = 28, PIN_OLFA_R_S1 = 29, PIN_OLFA_R_S2 = 30;
constexpr int PIN_SW_R_S    = 31;

// GLOBAL LOAD_REQ (from DAQ)
constexpr int PIN_GLOBAL_LOAD = 22;   // <-- set to your chosen input pin

// -------------------- SPI config --------------------
constexpr uint32_t SPI_HZ = 10000000;   // 10 MHz, safe for TPIC6B595

// -------------------- State tables --------------------
enum : uint8_t { ST_OFF=0, ST_AIR, ST_ODOR1, ST_ODOR2, ST_ODOR3, ST_ODOR4, ST_ODOR5, ST_FLUSH };

constexpr uint16_t OLFACTOMETER_STATES[8] = {
  /* OFF   */ 0x0000,
  /* AIR   */ BIT(0) | BIT(1),
  /* ODOR1 */ BIT(2) | BIT(3),
  /* ODOR2 */ BIT(4) | BIT(5),
  /* ODOR3 */ BIT(6) | BIT(7),
  /* ODOR4 */ BIT(8) | BIT(9),
  /* ODOR5 */ BIT(10) | BIT(11),
  /* FLUSH */ (uint16_t)0x0FFF
};

constexpr uint8_t SWITCH_STATES_2LVL[2] = {
  /* CLEAN */ 0b00000000,
  /* ODOR  */ 0b00000011
};

// Use identical tables for left/right; customize here if hardware differs
constexpr uint16_t OLFA_L_TAB[8] = {
  OLFACTOMETER_STATES[0], OLFACTOMETER_STATES[1], OLFACTOMETER_STATES[2], OLFACTOMETER_STATES[3],
  OLFACTOMETER_STATES[4], OLFACTOMETER_STATES[5], OLFACTOMETER_STATES[6], OLFACTOMETER_STATES[7]
};
constexpr uint16_t OLFA_R_TAB[8] = {
  OLFACTOMETER_STATES[0], OLFACTOMETER_STATES[1], OLFACTOMETER_STATES[2], OLFACTOMETER_STATES[3],
  OLFACTOMETER_STATES[4], OLFACTOMETER_STATES[5], OLFACTOMETER_STATES[6], OLFACTOMETER_STATES[7]
};
constexpr uint8_t  SW_L_TAB[2] = { SWITCH_STATES_2LVL[0], SWITCH_STATES_2LVL[1] };
constexpr uint8_t  SW_R_TAB[2] = { SWITCH_STATES_2LVL[0], SWITCH_STATES_2LVL[1] };

// -------------------- READY flags --------------------
volatile bool ready_olfa_l=false, ready_olfa_r=false, ready_sw_l=false, ready_sw_r=false;

// -------------------- SPI helpers --------------------
inline void spiShift16(uint16_t v) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer16(v);
  SPI.endTransaction();
}
inline void spiShift8(uint8_t v) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer(v);
  SPI.endTransaction();
}

// -------------------- GLOBAL LOAD ISR --------------------
void isr_global_load() {
  // Read S-lines (fast)
  uint8_t olfa_l_idx = (digitalReadFast(PIN_OLFA_L_S2) << 2)
                     | (digitalReadFast(PIN_OLFA_L_S1) << 1)
                     |  digitalReadFast(PIN_OLFA_L_S0);
  uint8_t olfa_r_idx = (digitalReadFast(PIN_OLFA_R_S2) << 2)
                     | (digitalReadFast(PIN_OLFA_R_S1) << 1)
                     |  digitalReadFast(PIN_OLFA_R_S0);
  uint8_t sw_l_idx = (digitalReadFast(PIN_SW_L_S) & 0x01);
  uint8_t sw_r_idx = (digitalReadFast(PIN_SW_R_S) & 0x01);

  // Build and shift entire 48-bit frame in chain order:
  // [OLFA_LEFT 16] -> [OLFA_RIGHT 16] -> [SW_LEFT 8] -> [SW_RIGHT 8]
  spiShift16(OLFA_L_TAB[olfa_l_idx & 0x07]);
  spiShift16(OLFA_R_TAB[olfa_r_idx & 0x07]);
  spiShift8 (SW_L_TAB[sw_l_idx & 0x01]);
  spiShift8 (SW_R_TAB[sw_r_idx & 0x01]);

  // Assert ALL readies high (any RCK can now commit its block)
  ready_olfa_l = ready_olfa_r = ready_sw_l = ready_sw_r = true;
  digitalWriteFast(PIN_READY_OLFA_L, HIGH);
  digitalWriteFast(PIN_READY_OLFA_R, HIGH);
  digitalWriteFast(PIN_READY_SW_L,   HIGH);
  digitalWriteFast(PIN_READY_SW_R,   HIGH);
}

// -------------------- RCK sense ISRs --------------------
void isr_rck_olfa_l() { if (ready_olfa_l) { ready_olfa_l = false; digitalWriteFast(PIN_READY_OLFA_L, LOW); } }
void isr_rck_olfa_r() { if (ready_olfa_r) { ready_olfa_r = false; digitalWriteFast(PIN_READY_OLFA_R, LOW); } }
void isr_rck_sw_l()   { if (ready_sw_l)   { ready_sw_l   = false; digitalWriteFast(PIN_READY_SW_L,   LOW); } }
void isr_rck_sw_r()   { if (ready_sw_r)   { ready_sw_r   = false; digitalWriteFast(PIN_READY_SW_R,   LOW); } }

// -------------------- Setup / Loop --------------------
void setup() {
  // SPI
  pinMode(10, OUTPUT); // keep CS as output per SPI lib
  SPI.begin();

  // Ready outs
  pinMode(PIN_READY_OLFA_L, OUTPUT); digitalWriteFast(PIN_READY_OLFA_L, LOW);
  pinMode(PIN_READY_OLFA_R, OUTPUT); digitalWriteFast(PIN_READY_OLFA_R, LOW);
  pinMode(PIN_READY_SW_L,   OUTPUT); digitalWriteFast(PIN_READY_SW_L,   LOW);
  pinMode(PIN_READY_SW_R,   OUTPUT); digitalWriteFast(PIN_READY_SW_R,   LOW);

  // S inputs
  pinMode(PIN_OLFA_L_S0, INPUT); pinMode(PIN_OLFA_L_S1, INPUT); pinMode(PIN_OLFA_L_S2, INPUT);
  pinMode(PIN_OLFA_R_S0, INPUT); pinMode(PIN_OLFA_R_S1, INPUT); pinMode(PIN_OLFA_R_S2, INPUT);
  pinMode(PIN_SW_L_S, INPUT);    pinMode(PIN_SW_R_S, INPUT);

  // Global LOAD input
  pinMode(PIN_GLOBAL_LOAD, INPUT);

  // RCK sense inputs
  pinMode(PIN_RCK_SENSE_OLFA_L, INPUT);
  pinMode(PIN_RCK_SENSE_OLFA_R, INPUT);
  pinMode(PIN_RCK_SENSE_SW_L,   INPUT);
  pinMode(PIN_RCK_SENSE_SW_R,   INPUT);

  // Interrupts
  attachInterrupt(digitalPinToInterrupt(PIN_GLOBAL_LOAD), isr_global_load, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLFA_L), isr_rck_olfa_l, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLFA_R), isr_rck_olfa_r, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SW_L),   isr_rck_sw_l,   RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SW_R),   isr_rck_sw_r,   RISING);

  // Preload known-safe OFF/CLEAN frame (DAQ can latch after arming)
  spiShift16(OLFA_L_TAB[ST_OFF]);
  spiShift16(OLFA_R_TAB[ST_OFF]);
  spiShift8 (SW_L_TAB[0]);
  spiShift8 (SW_R_TAB[0]);
}

void loop() {
  // Fully interrupt-driven
}