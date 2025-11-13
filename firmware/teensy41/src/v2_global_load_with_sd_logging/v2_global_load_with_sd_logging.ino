#include <Arduino.h>
#include <SPI.h>
#include <SD.h>

#ifndef BIT
#define BIT(n) (1u << (n))
#endif

// -------- SPI --------
constexpr uint32_t SPI_HZ = 1'000'000;
constexpr int PIN_CS = 10;  // Teensy SPI requirement

// -------- Your wiring --------
// S-lines (Teensy pins 0..7)
constexpr int PIN_S_OLF1_S1 = 0;
constexpr int PIN_S_SV1_S0  = 1;
constexpr int PIN_S_OLF2_S1 = 2;
constexpr int PIN_S_SV2_S0  = 3;
constexpr int PIN_S_OLF2_S2 = 4;
constexpr int PIN_S_OLF2_S0 = 5;
constexpr int PIN_S_OLF1_S2 = 6;
constexpr int PIN_S_OLF1_S0 = 7;

// GLOBAL LOAD (from DAQ)
constexpr int PIN_GLOBAL_LOAD = 23;

// RCK sense (from DAQ)
constexpr int PIN_RCK_SENSE_OLF1 = 19;
constexpr int PIN_RCK_SENSE_OLF2 = 20;
constexpr int PIN_RCK_SENSE_SV2  = 21;
constexpr int PIN_RCK_SENSE_SV1  = 22;

// READY outs (to DAQ)
constexpr int PIN_READY_OLF1 = 24;
constexpr int PIN_READY_SV1  = 25;
constexpr int PIN_READY_SV2  = 26;
constexpr int PIN_READY_OLF2 = 27;

// -------- SD logging --------
File logFile;
const int LOG_BUFFER_SIZE = 64;
volatile char logBuffer[LOG_BUFFER_SIZE][128];
volatile int logHead = 0, logTail = 0;

// -------- States / lookup --------
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

// READY flags
volatile bool ready_olf1=false, ready_olf2=false, ready_sv1=false, ready_sv2=false;

// -------- 48-bit frame builder --------
// Logical slots for the 6 bytes in the frame (NOT the send order)
enum FrameSlot : uint8_t { SLOT_OLF1_HI=0, SLOT_OLF1_LO, SLOT_OLF2_HI, SLOT_OLF2_LO, SLOT_SV1, SLOT_SV2 };

// Map logical slots to physical daisy-chain order (index 5 sent first → farthest TPIC)
uint8_t FRAME_SEND_ORDER[6] = {
  SLOT_OLF1_LO,  // out[0] → nearest (OLF1_LO)
  SLOT_OLF1_HI,  // out[1] → OLF1_HI
  SLOT_OLF2_LO,  // out[2] → OLF2_LO
  SLOT_OLF2_HI,  // out[3] → OLF2_HI
  SLOT_SV2,      // out[4] → SV2
  SLOT_SV1       // out[5] → farthest (SV1)
};
// If LEDs land on the wrong boards, reorder FRAME_SEND_ORDER (keep length 6).

static inline void build_frame(uint8_t out[6]) {
  // Read S-lines (fast)
  uint8_t olf1_idx = ((digitalReadFast(PIN_S_OLF1_S2) & 1) << 2) |
                     ((digitalReadFast(PIN_S_OLF1_S1) & 1) << 1) |
                     ((digitalReadFast(PIN_S_OLF1_S0) & 1) << 0);
  uint8_t olf2_idx = ((digitalReadFast(PIN_S_OLF2_S2) & 1) << 2) |
                     ((digitalReadFast(PIN_S_OLF2_S1) & 1) << 1) |
                     ((digitalReadFast(PIN_S_OLF2_S0) & 1) << 0);
  uint8_t sv1_idx  = (digitalReadFast(PIN_S_SV1_S0) & 1);
  uint8_t sv2_idx  = (digitalReadFast(PIN_S_SV2_S0) & 1);

  uint16_t olf1_val = OLFACTOMETER_STATES[olf1_idx & 0x07];
  uint16_t olf2_val = OLFACTOMETER_STATES[olf2_idx & 0x07];
  uint8_t  sv1_val  = SWITCH_STATES_2LVL[sv1_idx & 0x01];
  uint8_t  sv2_val  = SWITCH_STATES_2LVL[sv2_idx & 0x01];

  // Fill logical slots
  uint8_t slots[6];
  slots[SLOT_OLF1_HI] = (uint8_t)((olf1_val >> 8) & 0xFF);
  slots[SLOT_OLF1_LO] = (uint8_t)( olf1_val       & 0xFF);
  slots[SLOT_OLF2_HI] = (uint8_t)((olf2_val >> 8) & 0xFF);
  slots[SLOT_OLF2_LO] = (uint8_t)( olf2_val       & 0xFF);
  slots[SLOT_SV1]     = sv1_val;
  slots[SLOT_SV2]     = sv2_val;

  // Map to output buffer according to daisy-chain order (far → near)
  for (int i = 0; i < 6; ++i) out[i] = slots[FRAME_SEND_ORDER[i]];

  // Log LOAD
  int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
  if (nextHead != logTail) {
    unsigned long t = micros();
    snprintf((char*)logBuffer[logHead], 128,
      "%lu,LOAD,OLF1:%u,OLF2:%u,SV1:%u,SV2:%u,OLF1:%04X,OLF2:%04X,SV1:%02X,SV2:%02X",
      t, olf1_idx, olf2_idx, sv1_idx, sv2_idx, olf1_val, olf2_val, sv1_val, sv2_val);
    logHead = nextHead;
  }
}

static inline void spi_send_48(const uint8_t out[6]) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  // Send out[5] first (to farthest TPIC) ... out[0] last (nearest TPIC)
  for (int i = 5; i >= 0; --i) SPI.transfer(out[i]);
  SPI.endTransaction();
}

// -------- ISRs --------
void isr_global_load() {
  uint8_t bytes[6];
  build_frame(bytes);
  spi_send_48(bytes);

  ready_olf1 = ready_olf2 = ready_sv1 = ready_sv2 = true;
  digitalWriteFast(PIN_READY_OLF1, HIGH);
  digitalWriteFast(PIN_READY_OLF2, HIGH);
  digitalWriteFast(PIN_READY_SV1,  HIGH);
  digitalWriteFast(PIN_READY_SV2,  HIGH);
}

void isr_rck_olf1() {
  if (ready_olf1) {
    ready_olf1 = false;
    digitalWriteFast(PIN_READY_OLF1, LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) { snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,OLF1", micros()); logHead = nextHead; }
  }
}
void isr_rck_olf2() {
  if (ready_olf2) {
    ready_olf2 = false;
    digitalWriteFast(PIN_READY_OLF2, LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) { snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,OLF2", micros()); logHead = nextHead; }
  }
}
void isr_rck_sv2() {
  if (ready_sv2) {
    ready_sv2 = false;
    digitalWriteFast(PIN_READY_SV2, LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) { snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,SV2", micros()); logHead = nextHead; }
  }
}
void isr_rck_sv1() {
  if (ready_sv1) {
    ready_sv1 = false;
    digitalWriteFast(PIN_READY_SV1, LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) { snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,SV1", micros()); logHead = nextHead; }
  }
}

// -------- Setup / Loop --------
void setup() {
  Serial.begin(115200);
  delay(500);

  // SD card
  Serial.print("Initializing SD card...");
  if (!SD.begin(BUILTIN_SDCARD)) {
    Serial.println(" init failed!");
    while (1) {}
  }
  Serial.println(" done.");

  // Unique filename
  char logFileName[] = "log_000.txt";
  for (int i = 0; i < 1000; i++) {
    logFileName[4] = i / 100 + '0';
    logFileName[5] = (i / 10) % 10 + '0';
    logFileName[6] = i % 10 + '0';
    if (!SD.exists(logFileName)) break;
  }
  logFile = SD.open(logFileName, FILE_WRITE);
  if (logFile) {
    Serial.print("Logging to "); Serial.println(logFileName);
    logFile.println("--- MultiBiOS Log Start ---");
    logFile.println("Timestamp (us),Event,Details");
    logFile.flush();
  } else {
    Serial.println("Error opening log file!");
  }

  // SPI enable (Teensy requirement)
  pinMode(PIN_CS, OUTPUT);
  SPI.begin();

  // READY outs
  pinMode(PIN_READY_OLF1, OUTPUT); digitalWriteFast(PIN_READY_OLF1, LOW);
  pinMode(PIN_READY_SV1,  OUTPUT); digitalWriteFast(PIN_READY_SV1,  LOW);
  pinMode(PIN_READY_SV2,  OUTPUT); digitalWriteFast(PIN_READY_SV2,  LOW);
  pinMode(PIN_READY_OLF2, OUTPUT); digitalWriteFast(PIN_READY_OLF2, LOW);

  // S inputs
  pinMode(PIN_S_OLF1_S1, INPUT);
  pinMode(PIN_S_SV1_S0,  INPUT);
  pinMode(PIN_S_OLF2_S1, INPUT);
  pinMode(PIN_S_SV2_S0,  INPUT);
  pinMode(PIN_S_OLF2_S2, INPUT);
  pinMode(PIN_S_OLF2_S0, INPUT);
  pinMode(PIN_S_OLF1_S2, INPUT);
  pinMode(PIN_S_OLF1_S0, INPUT);

  // Global LOAD + RCK sense
  pinMode(PIN_GLOBAL_LOAD, INPUT);
  pinMode(PIN_RCK_SENSE_OLF1, INPUT);
  pinMode(PIN_RCK_SENSE_OLF2, INPUT);
  pinMode(PIN_RCK_SENSE_SV2,  INPUT);
  pinMode(PIN_RCK_SENSE_SV1,  INPUT);

  // Interrupts
  attachInterrupt(digitalPinToInterrupt(PIN_GLOBAL_LOAD),     isr_global_load, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLF1),  isr_rck_olf1,    RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLF2),  isr_rck_olf2,    RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SV2),   isr_rck_sv2,     RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SV1),   isr_rck_sv1,     RISING);

  // Preload OFF/CLEAN frame (outputs known before first latch)
  uint8_t initBytes[6] = {};
  spi_send_48(initBytes);
}

void loop() {
  // Drain log buffer safely
  if (logHead != logTail) {
    noInterrupts();
    int idx = logTail;
    interrupts();

    if (logFile) {
      logFile.println((char*)logBuffer[idx]);
      logFile.flush();
    }

    noInterrupts();
    logTail = (logTail + 1) % LOG_BUFFER_SIZE;
    interrupts();
  }
}
