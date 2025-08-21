/*
  Teensy 4.1 -> SHIFT REGISTERS (DAQ-controlled commit)
  ------------------------------------------------
  - Shared MOSI/SCK from Teensy to all shift register chains (no latching here)
  - NI-DAQ drives each chain's RCK (register clock) and timestamps that edge
  - Teensy preloads requested pattern upon *_LOAD_REQ, then asserts *_READY
  - When DAQ later pulses RCK_*, Teensy senses the edge, drops READY, unlocks SPI

  Electrical assumptions:
  - SHIFT REGISTER outputs sink current; a stored '1' turns output ON (OE\ low). (TI datasheet)
  - SRCLR\ tied HIGH, OE\ tied LOW on each shift register
  - Teensy I/O is 3.3V only (NOT 5V tolerant): level-shift/isolate DAQ -> Teensy inputs

  Pin plan (change if you like):
    SPI:    MOSI=11, SCK=13  (hardware SPI0)
    RCK sense inputs:   A=2,  B=3,  C=4,  D=5   (connected to the same DAQ-driven RCK nets)
    READY outputs:      A=6,  B=7,  C=8,  D=9
    State inputs:       A_S0=14, A_S1=15, A_S2=16
                        B_S0=17, B_S1=18, B_S2=19
                        C_S=20,  D_S=21
    LOAD_REQ inputs:    A=22, B=23, C=24, D=25
*/

#include <Arduino.h>
#include <SPI.h>

// -------------------- Pins --------------------
constexpr int PIN_MOSI = 11;
constexpr int PIN_SCK  = 13;

constexpr int PIN_RCK_SENSE_A = 2;
constexpr int PIN_RCK_SENSE_B = 3;
constexpr int PIN_RCK_SENSE_C = 4;
constexpr int PIN_RCK_SENSE_D = 5;

constexpr int PIN_A_READY = 6;
constexpr int PIN_B_READY = 7;
constexpr int PIN_C_READY = 8;
constexpr int PIN_D_READY = 9;

// Big A 3-bit state + load
constexpr int PIN_A_S0 = 14, PIN_A_S1 = 15, PIN_A_S2 = 16;
constexpr int PIN_A_LOAD = 22;

// Big B 3-bit state + load
constexpr int PIN_B_S0 = 17, PIN_B_S1 = 18, PIN_B_S2 = 19;
constexpr int PIN_B_LOAD = 23;

// Small C 1-bit state + load
constexpr int PIN_C_S = 20;
constexpr int PIN_C_LOAD = 24;

// Small D 1-bit state + load
constexpr int PIN_D_S = 21;
constexpr int PIN_D_LOAD = 25;

// -------------------- SPI config --------------------
constexpr uint32_t SPI_HZ = 10000000;  // 10 MHz: safe for TPIC6B595
// SPI mode 0, MSB-first is standard and fine for TPICs.

// -------------------- Helper macros --------------------
#ifndef BIT
#define BIT(n) (1u << (n))
#endif

// -------------------- State tables (bit n = valve v_n) --------------------
// Big: 16-bit word, using v0..v11 (v12..v15 spare)
enum : uint8_t { ST_OFF=0, ST_AIR, ST_ODOR1, ST_ODOR2, ST_ODOR3, ST_ODOR4, ST_ODOR5, ST_FLUSH };

constexpr uint16_t BIG_STATES[8] = {
  /* OFF   */ 0x0000,
  /* AIR   */ BIT(0) | BIT(1),
  /* ODOR1 */ BIT(2) | BIT(3),
  /* ODOR2 */ BIT(4) | BIT(5),
  /* ODOR3 */ BIT(6) | BIT(7),
  /* ODOR4 */ BIT(8) | BIT(9),
  /* ODOR5 */ BIT(10) | BIT(11),
  /* FLUSH */ (uint16_t)0x0FFF  // v0..v11 = 1
};

// Small: 8-bit word, using v0..v1 (rest spare)
constexpr uint8_t SMALL_STATES_2LEVEL[2] = {
  /* CLEAN */ 0b00000000,         // both OFF
  /* ODOR  */ 0b00000011          // v0 & v1 ON
};

// If both big manifolds share identical state patterns, use BIG_STATES for A & B.
// If they differ physically, copy and edit one array specifically for B:
constexpr uint16_t A_STATES[8] = {
  BIG_STATES[0], BIG_STATES[1], BIG_STATES[2], BIG_STATES[3],
  BIG_STATES[4], BIG_STATES[5], BIG_STATES[6], BIG_STATES[7]
};
constexpr uint16_t B_STATES[8] = {
  BIG_STATES[0], BIG_STATES[1], BIG_STATES[2], BIG_STATES[3],
  BIG_STATES[4], BIG_STATES[5], BIG_STATES[6], BIG_STATES[7]
};
constexpr uint8_t  C_STATES[2] = { SMALL_STATES_2LEVEL[0], SMALL_STATES_2LEVEL[1] };
constexpr uint8_t  D_STATES[2] = { SMALL_STATES_2LEVEL[0], SMALL_STATES_2LEVEL[1] };

// -------------------- Handshake state --------------------
enum Owner : uint8_t { OWNER_NONE=0, OWNER_A, OWNER_B, OWNER_C, OWNER_D };
volatile Owner busOwner = OWNER_NONE;    // who currently has a preloaded-but-unlatched pattern
volatile bool readyA=false, readyB=false, readyC=false, readyD=false;

// -------------------- SPI helpers --------------------
inline void spiShift16(uint16_t v) {
  // TPIC6B595: '1' in storage register turns output ON when latched and OE\ low. No inversion needed.
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer16(v);    // sends high byte then low byte (MSB-first)
  SPI.endTransaction();
}
inline void spiShift8(uint8_t v) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer(v);
  SPI.endTransaction();
}

// -------------------- LOAD_REQ ISRs: Preload only, assert READY --------------------
void isr_load_A() {
  if (busOwner != OWNER_NONE) return;           // a different preload is pending
  uint8_t idx = (digitalReadFast(PIN_A_S2) << 2)
              | (digitalReadFast(PIN_A_S1) << 1)
              |  digitalReadFast(PIN_A_S0);
  spiShift16(A_STATES[idx & 0x07]);             // preload (no latch)
  readyA = true; busOwner = OWNER_A;
  digitalWriteFast(PIN_A_READY, HIGH);
}
void isr_load_B() {
  if (busOwner != OWNER_NONE) return;
  uint8_t idx = (digitalReadFast(PIN_B_S2) << 2)
              | (digitalReadFast(PIN_B_S1) << 1)
              |  digitalReadFast(PIN_B_S0);
  spiShift16(B_STATES[idx & 0x07]);
  readyB = true; busOwner = OWNER_B;
  digitalWriteFast(PIN_B_READY, HIGH);
}
void isr_load_C() {
  if (busOwner != OWNER_NONE) return;
  uint8_t idx = digitalReadFast(PIN_C_S);
  spiShift8(C_STATES[idx & 0x01]);
  readyC = true; busOwner = OWNER_C;
  digitalWriteFast(PIN_C_READY, HIGH);
}
void isr_load_D() {
  if (busOwner != OWNER_NONE) return;
  uint8_t idx = digitalReadFast(PIN_D_S);
  spiShift8(D_STATES[idx & 0x01]);
  readyD = true; busOwner = OWNER_D;
  digitalWriteFast(PIN_D_READY, HIGH);
}

// -------------------- RCK sense ISRs: Commit happened (DAQ edge), drop READY --------------------
void isr_rck_A() {
  if (busOwner == OWNER_A && readyA) {
    readyA = false; busOwner = OWNER_NONE;
    digitalWriteFast(PIN_A_READY, LOW);
  }
}
void isr_rck_B() {
  if (busOwner == OWNER_B && readyB) {
    readyB = false; busOwner = OWNER_NONE;
    digitalWriteFast(PIN_B_READY, LOW);
  }
}
void isr_rck_C() {
  if (busOwner == OWNER_C && readyC) {
    readyC = false; busOwner = OWNER_NONE;
    digitalWriteFast(PIN_C_READY, LOW);
  }
}
void isr_rck_D() {
  if (busOwner == OWNER_D && readyD) {
    readyD = false; busOwner = OWNER_NONE;
    digitalWriteFast(PIN_D_READY, LOW);
  }
}

void setup() {
  // SPI
  pinMode(10, OUTPUT); // keep hardware CS as output per SPI library convention
  SPI.begin();         // MOSI=11, SCK=13 on Teensy 4.1 hardware SPI

  // READY outputs
  pinMode(PIN_A_READY, OUTPUT); digitalWriteFast(PIN_A_READY, LOW);
  pinMode(PIN_B_READY, OUTPUT); digitalWriteFast(PIN_B_READY, LOW);
  pinMode(PIN_C_READY, OUTPUT); digitalWriteFast(PIN_C_READY, LOW);
  pinMode(PIN_D_READY, OUTPUT); digitalWriteFast(PIN_D_READY, LOW);

  // State inputs from DAQ
  pinMode(PIN_A_S0, INPUT); pinMode(PIN_A_S1, INPUT); pinMode(PIN_A_S2, INPUT);
  pinMode(PIN_B_S0, INPUT); pinMode(PIN_B_S1, INPUT); pinMode(PIN_B_S2, INPUT);
  pinMode(PIN_C_S,  INPUT);
  pinMode(PIN_D_S,  INPUT);

  // LOAD_REQ inputs
  pinMode(PIN_A_LOAD, INPUT);
  pinMode(PIN_B_LOAD, INPUT);
  pinMode(PIN_C_LOAD, INPUT);
  pinMode(PIN_D_LOAD, INPUT);

  // RCK sense inputs (from DAQ’s RCK lines)
  pinMode(PIN_RCK_SENSE_A, INPUT);
  pinMode(PIN_RCK_SENSE_B, INPUT);
  pinMode(PIN_RCK_SENSE_C, INPUT);
  pinMode(PIN_RCK_SENSE_D, INPUT);

  // Attach interrupts
  attachInterrupt(digitalPinToInterrupt(PIN_A_LOAD), isr_load_A, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_B_LOAD), isr_load_B, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_C_LOAD), isr_load_C, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_D_LOAD), isr_load_D, RISING);

  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_A), isr_rck_A, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_B), isr_rck_B, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_C), isr_rck_C, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_D), isr_rck_D, RISING);

  // (Optional) Preload all to OFF so DAQ can latch them once everything is armed.
  spiShift16(A_STATES[ST_OFF]);
  spiShift16(B_STATES[ST_OFF]);
  spiShift8 (C_STATES[0]);
  spiShift8 (D_STATES[0]);

  // No READY asserted yet (not waiting for a latch). DAQ may make the first LOAD_REQ anytime.
}

void loop() {
  // Idle — all real-time work is interrupt-driven.
}
