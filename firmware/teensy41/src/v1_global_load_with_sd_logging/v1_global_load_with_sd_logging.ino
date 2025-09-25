/**
 * @file v1_global_load_with_sd_logging.ino
 * @brief MultiBiOS Teensy 4.1 firmware with SD card logging.
 * @version 1.1
 *
 * This version adds logging of all received commands and latching events to the
 * built-in microSD card on the Teensy 4.1.
 *
 * Wiring summary (example; match to your rig):
 *   SPI0: MOSI=11, SCK=13  (OE\ tied LOW, SRCLR\ tied HIGH on all TPIC6B595)
 *   SD Card: Uses the built-in microSD slot on the Teensy 4.1
 *   GLOBAL_LOAD_REQ  -> Teensy pin 22  (change below if needed)
 *   RCK sense (from DAQ): Left=A=2, Right=B=3, SwitchL=C=4, SwitchR=D=5
 *   READY out to DAQ: OLFA_L=6, OLFA_R=7, SW_L=8, SW_R=9
 *   S-lines from DAQ:
 *     Left  olfactometer: S0=14, S1=15, S2=16
 *     Right olfactometer: S0=17, S1=18, S2=19
 *     Left/Right switch:  S =20 / 21
 */

#include <Arduino.h>
#include <SPI.h>
#include <SD.h>

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
constexpr int PIN_GLOBAL_LOAD = 22;

// -------------------- SD Logging Config --------------------
File logFile;
const int LOG_BUFFER_SIZE = 64; // Number of log entries to buffer
volatile char logBuffer[LOG_BUFFER_SIZE][128]; // Buffer for log messages
volatile int logHead = 0; // Index to write to
volatile int logTail = 0; // Index to read from

// -------------------- SPI config --------------------
constexpr uint32_t SPI_HZ = 10000000;

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

constexpr uint16_t OLFA_L_TAB[8] = { OLFACTOMETER_STATES[0], OLFACTOMETER_STATES[1], OLFACTOMETER_STATES[2], OLFACTOMETER_STATES[3], OLFACTOMETER_STATES[4], OLFACTOMETER_STATES[5], OLFACTOMETER_STATES[6], OLFACTOMETER_STATES[7] };
constexpr uint16_t OLFA_R_TAB[8] = { OLFACTOMETER_STATES[0], OLFACTOMETER_STATES[1], OLFACTOMETER_STATES[2], OLFACTOMETER_STATES[3], OLFACTOMETER_STATES[4], OLFACTOMETER_STATES[5], OLFACTOMETER_STATES[6], OLFACTOMETER_STATES[7] };
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
  unsigned long timestamp = micros();

  // Read S-lines (fast)
  uint8_t olfa_l_idx = (digitalReadFast(PIN_OLFA_L_S2) << 2) | (digitalReadFast(PIN_OLFA_L_S1) << 1) | digitalReadFast(PIN_OLFA_L_S0);
  uint8_t olfa_r_idx = (digitalReadFast(PIN_OLFA_R_S2) << 2) | (digitalReadFast(PIN_OLFA_R_S1) << 1) | digitalReadFast(PIN_OLFA_R_S0);
  uint8_t sw_l_idx = (digitalReadFast(PIN_SW_L_S) & 0x01);
  uint8_t sw_r_idx = (digitalReadFast(PIN_SW_R_S) & 0x01);

  uint16_t olfa_l_val = OLFA_L_TAB[olfa_l_idx & 0x07];
  uint16_t olfa_r_val = OLFA_R_TAB[olfa_r_idx & 0x07];
  uint8_t  sw_l_val   = SW_L_TAB[sw_l_idx & 0x01];
  uint8_t  sw_r_val   = SW_R_TAB[sw_r_idx & 0x01];

  // Log the received command and the data to be sent
  int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
  if (nextHead != logTail) {
    snprintf((char*)logBuffer[logHead], 128, "%lu,LOAD,L_IDX:%d,R_IDX:%d,SWL_IDX:%d,SWR_IDX:%d,L_DATA:%04X,R_DATA:%04X,SWL_DATA:%02X,SWR_DATA:%02X",
             timestamp, olfa_l_idx, olfa_r_idx, sw_l_idx, sw_r_idx, olfa_l_val, olfa_r_val, sw_l_val, sw_r_val);
    logHead = nextHead;
  }

  // Build and shift entire 48-bit frame
  spiShift16(olfa_l_val);
  spiShift16(olfa_r_val);
  spiShift8(sw_l_val);
  spiShift8(sw_r_val);

  // Assert ALL readies high
  ready_olfa_l = ready_olfa_r = ready_sw_l = ready_sw_r = true;
  digitalWriteFast(PIN_READY_OLFA_L, HIGH);
  digitalWriteFast(PIN_READY_OLFA_R, HIGH);
  digitalWriteFast(PIN_READY_SW_L,   HIGH);
  digitalWriteFast(PIN_READY_SW_R,   HIGH);
}

// -------------------- RCK sense ISRs --------------------
void isr_rck_olfa_l() {
  if (ready_olfa_l) {
    ready_olfa_l = false;
    digitalWriteFast(PIN_READY_OLFA_L, LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) { snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,OLFA_L", micros()); logHead = nextHead; }
  }
}
void isr_rck_olfa_r() {
  if (ready_olfa_r) {
    ready_olfa_r = false;
    digitalWriteFast(PIN_READY_OLFA_R, LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) { snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,OLFA_R", micros()); logHead = nextHead; }
  }
}
void isr_rck_sw_l()   {
  if (ready_sw_l)   {
    ready_sw_l   = false;
    digitalWriteFast(PIN_READY_SW_L,   LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) { snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,SW_L", micros()); logHead = nextHead; }
  }
}
void isr_rck_sw_r()   {
  if (ready_sw_r)   {
    ready_sw_r   = false;
    digitalWriteFast(PIN_READY_SW_R,   LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) { snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,SW_R", micros()); logHead = nextHead; }
  }
}

// -------------------- Setup / Loop --------------------
void setup() {
  // Serial for debug
  Serial.begin(115200);
  delay(1000); // Wait for serial monitor to connect

  // SD Card Initialization
  Serial.print("Initializing SD card...");
  if (!SD.begin(BUILTIN_SDCARD)) {
    Serial.println(" initialization failed!");
    while (1); // Halt
  }
  Serial.println(" done.");

  // Find a unique log file name
  char logFileName[] = "log_000.txt";
  for (int i = 0; i < 1000; i++) {
    logFileName[4] = i / 100 + '0';
    logFileName[5] = (i / 10) % 10 + '0';
    logFileName[6] = i % 10 + '0';
    if (!SD.exists(logFileName)) {
      break;
    }
  }

  logFile = SD.open(logFileName, FILE_WRITE);
  if (logFile) {
    Serial.print("Logging to ");
    Serial.println(logFileName);
    logFile.println("--- MultiBiOS Log Start ---");
    logFile.println("Timestamp (us),Event,Details");
    logFile.flush();
  } else {
    Serial.println("Error opening log file!");
  }
  
  // SPI
  pinMode(10, OUTPUT);
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

  // Preload known-safe OFF/CLEAN frame
  spiShift16(OLFA_L_TAB[ST_OFF]);
  spiShift16(OLFA_R_TAB[ST_OFF]);
  spiShift8 (SW_L_TAB[0]);
  spiShift8 (SW_R_TAB[0]);
}

void loop() {
  // Check if there is data in the buffer to write to the SD card
  if (logHead != logTail) {
    if (logFile) {
      logFile.println((char*)logBuffer[logTail]);
      logFile.flush(); // Commit data to the card
    }
    logTail = (logTail + 1) % LOG_BUFFER_SIZE;
  }
}