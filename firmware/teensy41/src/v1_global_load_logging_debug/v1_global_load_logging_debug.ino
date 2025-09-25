/**
 * @file v1_global_load_logging_debug.ino
 * @brief Production firmware for the MultiBiOS Teensy 4.1 controller.
 * @version 1.3
 * @date 2025-09-22
 *
 * @mainpage MultiBiOS Teensy 4.1 Firmware
 *
 * @section intro_sec Introduction
 * This firmware provides a high-speed, interrupt-driven control solution for a dual
 * olfactometer and switch system. It is designed for real-time applications where
 * commands must be loaded and latched with minimal latency.
 *
 * @section features_sec Features
 *  - **Global Command Loading:** A single trigger (`GLOBAL_LOAD_REQ`) reads all command
 *    lines and shifts out a complete 48-bit SPI data frame.
 *  - **Independent Latching:** Each of the four hardware assemblies (left/right olfactometers,
 *    left/right switches) can latch its portion of the data frame independently via its own RCK signal.
 *  - **Non-Blocking SD Card Logging:** All events (command loads, latches) are timestamped
 *    with microsecond precision and logged to a microSD card without blocking critical interrupts.
 *  - **Real-time Status Monitoring:** A visual LED heartbeat indicates normal operation, while
 *    a detailed serial output provides startup diagnostics and live event reporting.
 *  - **Robust Error Handling:** The system halts and signals a clear error state if critical
 *    components (like the SD card) fail during initialization.
 *
 * @section wiring_sec Wiring Summary
 *  - **SPI0:** MOSI=11, SCK=13 (for the TPIC6B595 shift registers).
 *  - **SD Card:** Uses the built-in microSD slot on the Teensy 4.1.
 *  - **Control Signals (from DAQ):**
 *    - `GLOBAL_LOAD_REQ`: Pin 22
 *    - `RCK_SENSE_OLFA_L`: Pin 2
 *    - `RCK_SENSE_SW_L`:   Pin 3
 *    - `RCK_SENSE_OLFA_R`: Pin 4
 *    - `RCK_SENSE_SW_R`:   Pin 5
 *  - **Status Signals (to DAQ):**
 *    - `READY_OLFA_L`: Pin 6
 *    - `READY_SW_L`:   Pin 7
 *    - `READY_OLFA_R`: Pin 8
 *    - `READY_SW_R`:   Pin 9
 *  - **S-Line Inputs (Command from DAQ):**
 *    - Left Olfactometer: S0=24, S1=25, S2=26
 *    - Left Switch:       S=27
 *    - Right Olfactometer: S0=28, S1=29, S2=30
 *    - Right Switch:      S=31
 *
 * @section logic_sec Logical Flow
 * 1. **Initialization (`setup()`):**
 *    - Initializes Serial, SD card, SPI, and all I/O pins.
 *    - Creates a new, uniquely named log file (e.g., log_001.txt).
 *    - Preloads a known-safe "OFF" state into all shift registers.
 *    - Attaches all necessary interrupts.
 *    - Signals completion and enters the main loop.
 * 2. **Event Trigger (`isr_global_load()`):**
 *    - On a rising edge of `PIN_GLOBAL_LOAD`, this ISR executes.
 *    - It rapidly reads the state of all 8 S-line inputs.
 *    - It looks up the corresponding bit patterns from the state tables.
 *    - A log message is created and placed into a volatile buffer.
 *    - The full 48-bit data frame is shifted out via SPI.
 *    - All four `READY` lines are asserted HIGH.
 * 3. **Latching (`isr_rck_*()`):**
 *    - When a DAQ latches data for an assembly (e.g., Olfa_L), it raises the corresponding RCK line.
 *    - The associated `isr_rck_*()` function fires.
 *    - It logs the latch event to the buffer.
 *    - It de-asserts that assembly's `READY` line LOW, signaling completion.
 * 4. **Background Tasks (`loop()`):**
 *    - The main loop continuously runs two non-blocking tasks:
 *      a. **LED Heartbeat:** Blinks the onboard LED to show the system is alive.
 *      b. **SD Logging:** Checks the buffer for new log messages. If any exist, it
 *         writes them from the buffer to the SD card.
 */

//=================================================================================================
// LIBRARIES
//=================================================================================================
#include <Arduino.h>
#include <SPI.h>    // For shift register communication
#include <SD.h>     // For data logging

//=================================================================================================
// MACROS AND CONSTANTS
//=================================================================================================
#ifndef BIT
#define BIT(n) (1u << (n))
#endif

//=================================================================================================
// CONFIGURATION PARAMETERS
//=================================================================================================
// --- SPI Communication ---
constexpr uint32_t SPI_HZ = 10000000;   // 10 MHz. Safe for TPIC6B595 shift registers.

// --- Pin Definitions: Control & Status ---
constexpr int PIN_GLOBAL_LOAD = 22;     // INPUT:  Triggers reading S-lines and shifting data.
constexpr int PIN_LED_BUILTIN = 13;     // OUTPUT: Onboard LED for status indication.

// --- Pin Definitions: RCK Sense Inputs (from DAQ) ---
constexpr int PIN_RCK_SENSE_OLFA_L = 2;
constexpr int PIN_RCK_SENSE_SW_L   = 3;
constexpr int PIN_RCK_SENSE_OLFA_R = 4;
constexpr int PIN_RCK_SENSE_SW_R   = 5;

// --- Pin Definitions: READY Outputs (to DAQ) ---
constexpr int PIN_READY_OLFA_L = 6;
constexpr int PIN_READY_SW_L   = 7;
constexpr int PIN_READY_OLFA_R = 8;
constexpr int PIN_READY_SW_R   = 9;

// --- Pin Definitions: S-Line Command Inputs (from DAQ) ---
constexpr int PIN_OLFA_L_S0 = 24, PIN_OLFA_L_S1 = 25, PIN_OLFA_L_S2 = 26;
constexpr int PIN_SW_L_S    = 27;
constexpr int PIN_OLFA_R_S0 = 28, PIN_OLFA_R_S1 = 29, PIN_OLFA_R_S2 = 30;
constexpr int PIN_SW_R_S    = 31;

// --- SD Logging ---
const int LOG_BUFFER_SIZE = 64;         // Max number of log entries to buffer in RAM.
                                        // A larger buffer can handle higher event rates but uses more memory.

//=================================================================================================
// STATE MACHINE DEFINITIONS
//=================================================================================================
// These enums and arrays define the mapping from a simple command index (read from S-lines)
// to the specific 16-bit or 8-bit patterns that need to be sent to the hardware.

enum OlfactometerState : uint8_t { ST_OFF=0, ST_AIR, ST_ODOR1, ST_ODOR2, ST_ODOR3, ST_ODOR4, ST_ODOR5, ST_FLUSH };
enum SwitchState : uint8_t { SW_CLEAN=0, SW_ODOR };

// Base bit patterns for the olfactometer hardware.
constexpr uint16_t OLFACTOMETER_STATES[8] = {
  /* OFF   */ 0x0000,
  /* AIR   */ BIT(0) | BIT(1),
  /* ODOR1 */ BIT(2) | BIT(3),
  /* ODOR2 */ BIT(4) | BIT(5),
  /* ODOR3 */ BIT(6) | BIT(7),
  /* ODOR4 */ BIT(8) | BIT(9),
  /* ODOR5 */ BIT(10) | BIT(11),
  /* FLUSH */ (uint16_t)0x0FFF // Assumes 12 valves
};

// Base bit patterns for the 2-level switch hardware.
constexpr uint8_t SWITCH_STATES_2LVL[2] = {
  /* CLEAN */ 0b00000000,
  /* ODOR  */ 0b00000011
};

// Final lookup tables used by the ISR. Customize these if L/R hardware differs.
constexpr uint16_t OLFA_L_TAB[8] = { OLFACTOMETER_STATES[0], OLFACTOMETER_STATES[1], OLFACTOMETER_STATES[2], OLFACTOMETER_STATES[3], OLFACTOMETER_STATES[4], OLFACTOMETER_STATES[5], OLFACTOMETER_STATES[6], OLFACTOMETER_STATES[7] };
constexpr uint16_t OLFA_R_TAB[8] = { OLFACTOMETER_STATES[0], OLFACTOMETER_STATES[1], OLFACTOMETER_STATES[2], OLFACTOMETER_STATES[3], OLFACTOMETER_STATES[4], OLFACTOMETER_STATES[5], OLFACTOMETER_STATES[6], OLFACTOMETER_STATES[7] };
constexpr uint8_t  SW_L_TAB[2] = { SWITCH_STATES_2LVL[0], SWITCH_STATES_2LVL[1] };
constexpr uint8_t  SW_R_TAB[2] = { SWITCH_STATES_2LVL[0], SWITCH_STATES_2LVL[1] };


//=================================================================================================
// GLOBAL VARIABLES
//=================================================================================================
// --- SD Logging Buffer ---
// These variables are shared between ISRs and the main loop, so they MUST be 'volatile'.
// This prevents the compiler from making optimizations that could break the code.
volatile char logBuffer[LOG_BUFFER_SIZE][128]; // A circular buffer for log messages.
volatile int logHead = 0;                      // The index where the next message will be written by an ISR.
volatile int logTail = 0;                      // The index where the next message will be read by the main loop.

// --- Ready State Flags ---
// These are also 'volatile' as they are written in ISRs and read in other ISRs.
volatile bool ready_olfa_l = false;
volatile bool ready_olfa_r = false;
volatile bool ready_sw_l   = false;
volatile bool ready_sw_r   = false;

// --- File handle for SD card ---
File logFile;

//=================================================================================================
// HELPER FUNCTIONS
//=================================================================================================

/**
 * @brief Shifts out a 16-bit value via SPI.
 * @param v The 16-bit data to send.
 */
inline void spiShift16(uint16_t v) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer16(v);
  SPI.endTransaction();
}

/**
 * @brief Shifts out an 8-bit value via SPI.
 * @param v The 8-bit data to send.
 */
inline void spiShift8(uint8_t v) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer(v);
  SPI.endTransaction();
}

/**
 * @brief Signals a fatal error by blinking the onboard LED in an SOS pattern.
 *        This function halts all execution.
 */
void signal_error_led() {
  pinMode(PIN_LED_BUILTIN, OUTPUT);
  while(true) {
    // S (...): 3 short blinks
    for(int i=0; i<3; i++) { digitalWrite(PIN_LED_BUILTIN, HIGH); delay(150); digitalWrite(PIN_LED_BUILTIN, LOW); delay(100); }
    delay(250);
    // O (---): 3 long blinks
    for(int i=0; i<3; i++) { digitalWrite(PIN_LED_BUILTIN, HIGH); delay(400); digitalWrite(PIN_LED_BUILTIN, LOW); delay(100); }
    delay(250);
    // S (...): 3 short blinks
    for(int i=0; i<3; i++) { digitalWrite(PIN_LED_BUILTIN, HIGH); delay(150); digitalWrite(PIN_LED_BUILTIN, LOW); delay(100); }
    delay(1000); // Wait and repeat
  }
}

//=================================================================================================
// INTERRUPT SERVICE ROUTINES (ISRs)
//
// ! ! ! CRITICAL NOTE ! ! !
// ISRs must be as fast as possible. Avoid any long-running operations like Serial prints,
// SD card writes, or complex calculations. Here, we read pins, format a string in RAM,
// and perform a hardware SPI transfer, all of which are extremely fast operations.
//=================================================================================================

/**
 * @brief ISR for the GLOBAL_LOAD_REQ signal.
 *        Reads S-lines, logs the command, shifts out all 48 bits, and raises READY lines.
 */
void isr_global_load() {
  unsigned long timestamp = micros();

  // 1. Read all S-lines as quickly as possible.
  uint8_t olfa_l_idx = (digitalReadFast(PIN_OLFA_L_S2) << 2) | (digitalReadFast(PIN_OLFA_L_S1) << 1) | digitalReadFast(PIN_OLFA_L_S0);
  uint8_t olfa_r_idx = (digitalReadFast(PIN_OLFA_R_S2) << 2) | (digitalReadFast(PIN_OLFA_R_S1) << 1) | digitalReadFast(PIN_OLFA_R_S0);
  uint8_t sw_l_idx = (digitalReadFast(PIN_SW_L_S) & 0x01);
  uint8_t sw_r_idx = (digitalReadFast(PIN_SW_R_S) & 0x01);

  // 2. Look up the corresponding data patterns from the state tables.
  uint16_t olfa_l_val = OLFA_L_TAB[olfa_l_idx & 0x07];
  uint16_t olfa_r_val = OLFA_R_TAB[olfa_r_idx & 0x07];
  uint8_t  sw_l_val   = SW_L_TAB[sw_l_idx & 0x01];
  uint8_t  sw_r_val   = SW_R_TAB[sw_r_idx & 0x01];

  // 3. Queue the log message into the circular buffer.
  int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
  if (nextHead != logTail) { // Check for buffer overflow.
    snprintf((char*)logBuffer[logHead], 128, "%lu,LOAD,L_IDX:%d,R_IDX:%d,SWL_IDX:%d,SWR_IDX:%d,L_DATA:%04X,R_DATA:%04X,SWL_DATA:%02X,SWR_DATA:%02X",
             timestamp, olfa_l_idx, olfa_r_idx, sw_l_idx, sw_r_idx, olfa_l_val, olfa_r_val, sw_l_val, sw_r_val);
    logHead = nextHead;
  }
  // If the buffer is full, this message is dropped to prevent halting the system.

  // 4. Shift out the entire 48-bit frame in the correct hardware chain order.
  // Chain: [OLFA_LEFT 16b] -> [OLFA_RIGHT 16b] -> [SW_LEFT 8b] -> [SW_RIGHT 8b]
  spiShift16(olfa_l_val);
  spiShift16(olfa_r_val);
  spiShift8(sw_l_val);
  spiShift8(sw_r_val);

  // 5. Assert all READY lines HIGH to signal that data is ready to be latched.
  ready_olfa_l = ready_olfa_r = ready_sw_l = ready_sw_r = true;
  digitalWriteFast(PIN_READY_OLFA_L, HIGH);
  digitalWriteFast(PIN_READY_OLFA_R, HIGH);
  digitalWriteFast(PIN_READY_SW_L,   HIGH);
  digitalWriteFast(PIN_READY_SW_R,   HIGH);
}

/**
 * @brief ISR for the Left Olfactometer RCK signal. Logs the event and lowers the READY line.
 */
void isr_rck_olfa_l() {
  if (ready_olfa_l) { // Only act if a load was pending.
    ready_olfa_l = false;
    digitalWriteFast(PIN_READY_OLFA_L, LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) {
      snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,OLFA_L", micros());
      logHead = nextHead;
    }
  }
}

/**
 * @brief ISR for the Right Olfactometer RCK signal. Logs the event and lowers the READY line.
 */
void isr_rck_olfa_r() {
  if (ready_olfa_r) {
    ready_olfa_r = false;
    digitalWriteFast(PIN_READY_OLFA_R, LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) {
      snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,OLFA_R", micros());
      logHead = nextHead;
    }
  }
}

/**
 * @brief ISR for the Left Switch RCK signal. Logs the event and lowers the READY line.
 */
void isr_rck_sw_l() {
  if (ready_sw_l) {
    ready_sw_l = false;
    digitalWriteFast(PIN_READY_SW_L,   LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) {
      snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,SW_L", micros());
      logHead = nextHead;
    }
  }
}

/**
 * @brief ISR for the Right Switch RCK signal. Logs the event and lowers the READY line.
 */
void isr_rck_sw_r() {
  if (ready_sw_r) {
    ready_sw_r = false;
    digitalWriteFast(PIN_READY_SW_R,   LOW);
    int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
    if (nextHead != logTail) {
      snprintf((char*)logBuffer[logHead], 128, "%lu,RCK,SW_R", micros());
      logHead = nextHead;
    }
  }
}

//=================================================================================================
// SETUP
//=================================================================================================
/**
 * @brief Initializes all hardware and software components on startup.
 */
void setup() {
  pinMode(PIN_LED_BUILTIN, OUTPUT);
  digitalWrite(PIN_LED_BUILTIN, HIGH); // Turn LED on during setup as a visual cue.

  // Start Serial communication for debugging and status messages.
  Serial.begin(115200);
  delay(1000); // Allow time for Serial Monitor to connect.
  Serial.println("\n\n--- MultiBiOS Firmware v1.3 Booting Up ---");

  // Initialize the SD card using the Teensy 4.1's built-in slot.
  Serial.print("Initializing SD card...");
  if (!SD.begin(BUILTIN_SDCARD)) {
    Serial.println(" FATAL ERROR: SD Card initialization failed!");
    Serial.println("Check card formatting (FAT32/exFAT) and connection. System halted.");
    signal_error_led(); // Halt and blink SOS.
  }
  Serial.println(" OK.");

  // Find a unique log file name to avoid overwriting previous data.
  char logFileName[] = "log_000.txt";
  for (int i = 0; i < 1000; i++) {
    logFileName[4] = i / 100 + '0';
    logFileName[5] = (i / 10) % 10 + '0';
    logFileName[6] = i % 10 + '0';
    if (!SD.exists(logFileName)) {
      break;
    }
  }

  // Open the log file for writing.
  logFile = SD.open(logFileName, FILE_WRITE);
  if (logFile) {
    Serial.print("Successfully opened log file: ");
    Serial.println(logFileName);
    logFile.println("--- MultiBiOS Log Start ---");
    logFile.println("Timestamp (us),Event,Details");
    logFile.flush(); // Write header immediately.
  } else {
    Serial.print("FATAL ERROR: Could not create log file ");
    Serial.println(logFileName);
    signal_error_led(); // Halt and blink SOS.
  }

  // Initialize SPI communication.
  Serial.print("Configuring SPI and I/O pins...");
  pinMode(10, OUTPUT); 
  SPI.begin();

  // Configure READY lines as outputs and set to a known initial state (LOW).
  pinMode(PIN_READY_OLFA_L, OUTPUT); digitalWriteFast(PIN_READY_OLFA_L, LOW);
  pinMode(PIN_READY_OLFA_R, OUTPUT); digitalWriteFast(PIN_READY_OLFA_R, LOW);
  pinMode(PIN_READY_SW_L,   OUTPUT); digitalWriteFast(PIN_READY_SW_L,   LOW);
  pinMode(PIN_READY_SW_R,   OUTPUT); digitalWriteFast(PIN_READY_SW_R,   LOW);

  // Configure S-lines as inputs. (These do not need pull-downs as they are not interrupt pins).
  pinMode(PIN_OLFA_L_S0, INPUT); pinMode(PIN_OLFA_L_S1, INPUT); pinMode(PIN_OLFA_L_S2, INPUT);
  pinMode(PIN_OLFA_R_S0, INPUT); pinMode(PIN_OLFA_R_S1, INPUT); pinMode(PIN_OLFA_R_S2, INPUT);
  pinMode(PIN_SW_L_S, INPUT);    pinMode(PIN_SW_R_S, INPUT);

  // Configure GLOBAL_LOAD and RCK lines as inputs WITH PULL-DOWNS.
  // This is the critical fix to prevent floating inputs from triggering false interrupts.
  pinMode(PIN_GLOBAL_LOAD,      INPUT_PULLDOWN);
  pinMode(PIN_RCK_SENSE_OLFA_L, INPUT_PULLDOWN);
  pinMode(PIN_RCK_SENSE_OLFA_R, INPUT_PULLDOWN);
  pinMode(PIN_RCK_SENSE_SW_L,   INPUT_PULLDOWN);
  pinMode(PIN_RCK_SENSE_SW_R,   INPUT_PULLDOWN);
  Serial.println(" OK.");

  // Attach all interrupts.
  Serial.print("Attaching interrupts...");
  attachInterrupt(digitalPinToInterrupt(PIN_GLOBAL_LOAD), isr_global_load, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLFA_L), isr_rck_olfa_l, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLFA_R), isr_rck_olfa_r, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SW_L),   isr_rck_sw_l,   RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SW_R),   isr_rck_sw_r,   RISING);
  Serial.println(" OK.");

  // Preload the shift registers with a known-safe OFF/CLEAN state.
  Serial.print("Preloading safe initial state to shift registers...");
  spiShift16(OLFA_L_TAB[ST_OFF]);
  spiShift16(OLFA_R_TAB[ST_OFF]);
  spiShift8 (SW_L_TAB[SW_CLEAN]);
  spiShift8 (SW_R_TAB[SW_CLEAN]);
  Serial.println(" OK.");

  Serial.println("\n--- Setup Complete. System is active. ---");
  digitalWrite(PIN_LED_BUILTIN, LOW); // Turn LED off, ready for heartbeat.
}

//=================================================================================================
// MAIN LOOP
//=================================================================================================
/**
 * @brief The main loop handles non-time-critical background tasks.
 *        The system is primarily interrupt-driven.
 */
void loop() {
  // --- Task 1: Non-blocking LED Heartbeat ---
  // This provides a constant visual confirmation that the firmware has not crashed.
  static unsigned long lastHeartbeatTime = 0;
  if (millis() - lastHeartbeatTime > 500) { // Blink every 500 ms.
    lastHeartbeatTime = millis();
    digitalWrite(PIN_LED_BUILTIN, !digitalRead(PIN_LED_BUILTIN)); // Toggle LED state.
  }

  // --- Task 2: Process the SD Log Buffer ---
  // Check if the ISR has added any new messages to the buffer.
  if (logHead != logTail) {
    if (logFile) {
      // Write the oldest message from the buffer to the SD card.
      logFile.println((char*)logBuffer[logTail]);
      
      // Also print to Serial for live monitoring.
      Serial.print("Logged: ");
      Serial.println((char*)logBuffer[logTail]);
      
      // IMPORTANT: Flush the file to ensure data is physically written to the card.
      // This minimizes data loss in case of sudden power-off. For very high event rates,
      // you might flush less often (e.g., every N entries) to improve performance.
      logFile.flush();
    }
    // Advance the tail of the circular buffer.
    logTail = (logTail + 1) % LOG_BUFFER_SIZE;
  }
}