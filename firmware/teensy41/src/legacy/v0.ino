/**
 * @file v0.ino
 * @brief MultiBiOS Olfactometer Control Firmware for Teensy 4.1
 * @version 1.0
 * @date 2025
 * 
 * @description
 * This firmware controls 4 shift register chains for olfactometer valve control.
 * The system uses a DAQ-controlled commit mechanism where the Teensy preloads
 * valve patterns and the NI-DAQ provides precise timing for valve state changes.
 * 
 * @hardware_setup
 * - Teensy 4.1 microcontroller
 * - 4x TPIC6B595 shift register chains
 * - NI-DAQ for timing control
 * - Level shifters for 5V/3.3V compatibility
 * 
 * @operation_principle
 * 1. DAQ sets state bits and pulses LOAD_REQ
 * 2. Teensy reads state, preloads SPI pattern, asserts READY
 * 3. DAQ pulses RCK (register clock) to commit pattern
 * 4. Teensy senses RCK edge, drops READY, unlocks SPI bus
 * 
 * @electrical_notes
 * - SHIFT REGISTER outputs sink current; '1' bit turns valve ON (when OE\ low)
 * - SRCLR\ tied HIGH, OE\ tied LOW on all shift registers
 * - Teensy I/O is 3.3V only: use level shifters for DAQ interface
 * - All timing-critical operations are interrupt-driven
 */

#include <Arduino.h>
#include <SPI.h>

// =====================================================================================
// PIN DEFINITIONS
// =====================================================================================

/**
 * @section SPI_PINS Hardware SPI pins for shift register communication
 * Using Teensy 4.1 hardware SPI0 for maximum speed and reliability
 */
constexpr int PIN_MOSI = 11;  ///< Master Out Slave In - data to shift registers
constexpr int PIN_SCK  = 13;  ///< Serial Clock - shared clock for all chains

/**
 * @section RCK_SENSE_PINS Register Clock sense inputs from DAQ
 * These pins monitor when the DAQ pulses RCK to commit preloaded patterns
 */
constexpr int PIN_RCK_SENSE_OLFACTOMETER_LEFT  = 2;  ///< Left olfactometer RCK sense
constexpr int PIN_RCK_SENSE_OLFACTOMETER_RIGHT = 3;  ///< Right olfactometer RCK sense
constexpr int PIN_RCK_SENSE_SWITCHVALVE_LEFT   = 4;  ///< Left switchvalve RCK sense
constexpr int PIN_RCK_SENSE_SWITCHVALVE_RIGHT  = 5;  ///< Right switchvalve RCK sense

/**
 * @section READY_PINS Ready signal outputs to DAQ
 * These signals indicate when Teensy has preloaded a pattern and is ready for commit
 */
constexpr int PIN_OLFACTOMETER_LEFT_READY  = 6;  ///< Left olfactometer ready signal
constexpr int PIN_OLFACTOMETER_RIGHT_READY = 7;  ///< Right olfactometer ready signal
constexpr int PIN_SWITCHVALVE_LEFT_READY   = 8;  ///< Left switchvalve ready signal
constexpr int PIN_SWITCHVALVE_RIGHT_READY  = 9;  ///< Right switchvalve ready signal

/**
 * @section OLFACTOMETER_LEFT_PINS Left olfactometer state and control pins
 * 3-bit state input allows 8 different valve configurations
 */
constexpr int PIN_OLFACTOMETER_LEFT_S0   = 14;  ///< State bit 0 (LSB)
constexpr int PIN_OLFACTOMETER_LEFT_S1   = 15;  ///< State bit 1
constexpr int PIN_OLFACTOMETER_LEFT_S2   = 16;  ///< State bit 2 (MSB)
constexpr int PIN_OLFACTOMETER_LEFT_LOAD = 22;  ///< Load request trigger

/**
 * @section OLFACTOMETER_RIGHT_PINS Right olfactometer state and control pins
 * 3-bit state input allows 8 different valve configurations
 */
constexpr int PIN_OLFACTOMETER_RIGHT_S0   = 17;  ///< State bit 0 (LSB)
constexpr int PIN_OLFACTOMETER_RIGHT_S1   = 18;  ///< State bit 1
constexpr int PIN_OLFACTOMETER_RIGHT_S2   = 19;  ///< State bit 2 (MSB)
constexpr int PIN_OLFACTOMETER_RIGHT_LOAD = 23;  ///< Load request trigger

/**
 * @section SWITCHVALVE_LEFT_PINS Left switchvalve state and control pins
 * 1-bit state input allows 2 different valve configurations (clean/odor)
 */
constexpr int PIN_SWITCHVALVE_LEFT_S    = 20;  ///< State bit (0=clean, 1=odor)
constexpr int PIN_SWITCHVALVE_LEFT_LOAD = 24;  ///< Load request trigger

/**
 * @section SWITCHVALVE_RIGHT_PINS Right switchvalve state and control pins
 * 1-bit state input allows 2 different valve configurations (clean/odor)
 */
constexpr int PIN_SWITCHVALVE_RIGHT_S    = 21;  ///< State bit (0=clean, 1=odor)
constexpr int PIN_SWITCHVALVE_RIGHT_LOAD = 25;  ///< Load request trigger
// =====================================================================================
// SPI CONFIGURATION
// =====================================================================================

/**
 * @section SPI_CONFIG SPI communication parameters
 * Optimized for TPIC6B595 shift registers with safety margins
 */
constexpr uint32_t SPI_HZ = 10000000;  ///< 10 MHz SPI clock (safe for TPIC6B595)
// Note: Using SPI_MODE0 (CPOL=0, CPHA=0) and MSBFIRST as standard for shift registers

// =====================================================================================
// HELPER MACROS
// =====================================================================================

#ifndef BIT
/**
 * @brief Bit manipulation macro for creating bit masks
 * @param n Bit position (0-based)
 * @return Unsigned integer with bit n set
 */
#define BIT(n) (1u << (n))
#endif

// =====================================================================================
// VALVE STATE DEFINITIONS
// =====================================================================================

/**
 * @enum OlfactometerState
 * @brief Enumeration of possible olfactometer states
 * 
 * Each olfactometer can be in one of 8 states, controlled by 3-bit input.
 * These states determine which valves are opened for different odor delivery modes.
 */
enum OlfactometerState : uint8_t {
  ST_OFF   = 0,  ///< All valves closed
  ST_AIR   = 1,  ///< Air delivery (clean carrier)
  ST_ODOR1 = 2,  ///< Odor channel 1
  ST_ODOR2 = 3,  ///< Odor channel 2
  ST_ODOR3 = 4,  ///< Odor channel 3
  ST_ODOR4 = 5,  ///< Odor channel 4
  ST_ODOR5 = 6,  ///< Odor channel 5
  ST_FLUSH = 7   ///< Flush mode (all odor valves open for cleaning)
};

/**
 * @brief Olfactometer valve pattern lookup table
 * 
 * Maps state enum to 16-bit valve control pattern.
 * Bits 0-11 control valves v0-v11, bits 12-15 are spare.
 * 
 * Valve assignment:
 * - v0,v1: Air delivery valves
 * - v2,v3: Odor 1 valves
 * - v4,v5: Odor 2 valves
 * - etc.
 */
constexpr uint16_t OLFACTOMETER_STATES[8] = {
  /* ST_OFF   */ 0x0000,                    // All valves closed
  /* ST_AIR   */ BIT(0) | BIT(1),           // Air valves open
  /* ST_ODOR1 */ BIT(2) | BIT(3),           // Odor 1 valves open
  /* ST_ODOR2 */ BIT(4) | BIT(5),           // Odor 2 valves open
  /* ST_ODOR3 */ BIT(6) | BIT(7),           // Odor 3 valves open
  /* ST_ODOR4 */ BIT(8) | BIT(9),           // Odor 4 valves open
  /* ST_ODOR5 */ BIT(10) | BIT(11),         // Odor 5 valves open
  /* ST_FLUSH */ 0x0FFF                     // All odor valves open (v0-v11)
};

/**
 * @brief Switchvalve pattern lookup table
 * 
 * Maps 1-bit state to 8-bit valve control pattern.
 * Only uses bits 0-1 for valve control, rest are spare.
 * 
 * States:
 * - 0 (CLEAN): Both valves closed for clean air path
 * - 1 (ODOR):  Both valves open for odor delivery path
 */
constexpr uint8_t SWITCHVALVE_STATES_2LEVEL[2] = {
  /* CLEAN */ 0b00000000,         // Both valves closed
  /* ODOR  */ 0b00000011          // v0 & v1 open
};

// =====================================================================================
// DEVICE-SPECIFIC STATE TABLES
// =====================================================================================

/**
 * @brief Left olfactometer state lookup table
 * 
 * Individual state table allows for device-specific calibration if needed.
 * Currently identical to base OLFACTOMETER_STATES but can be customized
 * for left manifold characteristics.
 */
constexpr uint16_t OLFACTOMETER_LEFT_STATES[8] = {
  OLFACTOMETER_STATES[ST_OFF],   OLFACTOMETER_STATES[ST_AIR],
  OLFACTOMETER_STATES[ST_ODOR1], OLFACTOMETER_STATES[ST_ODOR2],
  OLFACTOMETER_STATES[ST_ODOR3], OLFACTOMETER_STATES[ST_ODOR4],
  OLFACTOMETER_STATES[ST_ODOR5], OLFACTOMETER_STATES[ST_FLUSH]
};

/**
 * @brief Right olfactometer state lookup table
 * 
 * Individual state table allows for device-specific calibration if needed.
 * Currently identical to base OLFACTOMETER_STATES but can be customized
 * for right manifold characteristics.
 */
constexpr uint16_t OLFACTOMETER_RIGHT_STATES[8] = {
  OLFACTOMETER_STATES[ST_OFF],   OLFACTOMETER_STATES[ST_AIR],
  OLFACTOMETER_STATES[ST_ODOR1], OLFACTOMETER_STATES[ST_ODOR2],
  OLFACTOMETER_STATES[ST_ODOR3], OLFACTOMETER_STATES[ST_ODOR4],
  OLFACTOMETER_STATES[ST_ODOR5], OLFACTOMETER_STATES[ST_FLUSH]
};

/**
 * @brief Left switchvalve state lookup table
 * Copy of base 2-level states for consistency and future customization
 */
constexpr uint8_t SWITCHVALVE_LEFT_STATES[2] = {
  SWITCHVALVE_STATES_2LEVEL[0], // CLEAN
  SWITCHVALVE_STATES_2LEVEL[1]  // ODOR
};

/**
 * @brief Right switchvalve state lookup table
 * Copy of base 2-level states for consistency and future customization
 */
constexpr uint8_t SWITCHVALVE_RIGHT_STATES[2] = {
  SWITCHVALVE_STATES_2LEVEL[0], // CLEAN
  SWITCHVALVE_STATES_2LEVEL[1]  // ODOR
};

// =====================================================================================
// HANDSHAKE STATE MANAGEMENT
// =====================================================================================

/**
 * @enum Owner
 * @brief Tracks which device currently owns the SPI bus
 * 
 * Only one device can have a preloaded pattern at a time to prevent
 * SPI bus conflicts. This enum tracks the current bus owner.
 */
enum Owner : uint8_t {
  OWNER_NONE = 0,               ///< No device owns the bus (idle state)
  OWNER_OLFACTOMETER_LEFT,      ///< Left olfactometer has preloaded pattern
  OWNER_OLFACTOMETER_RIGHT,     ///< Right olfactometer has preloaded pattern
  OWNER_SWITCHVALVE_LEFT,       ///< Left switchvalve has preloaded pattern
  OWNER_SWITCHVALVE_RIGHT       ///< Right switchvalve has preloaded pattern
};

/**
 * @brief Current SPI bus owner
 * 
 * Volatile because it's modified in interrupt service routines.
 * Tracks which device currently has a preloaded-but-uncommitted pattern.
 */
volatile Owner busOwner = OWNER_NONE;

/**
 * @brief Ready state flags for each device
 * 
 * These volatile flags track whether each device has successfully preloaded
 * a pattern and is ready for the DAQ to pulse RCK for commit.
 * All flags are modified in interrupt service routines.
 */
volatile bool readyOlfactometerLeft  = false;  ///< Left olfactometer ready flag
volatile bool readyOlfactometerRight = false;  ///< Right olfactometer ready flag
volatile bool readySwitchvalveLeft   = false;  ///< Left switchvalve ready flag
volatile bool readySwitchvalveRight  = false;  ///< Right switchvalve ready flag

// =====================================================================================
// SPI COMMUNICATION HELPERS
// =====================================================================================

/**
 * @brief Send 16-bit data via SPI to shift register chain
 * @param value 16-bit pattern to send (MSB first)
 * 
 * Configures SPI transaction and sends 16-bit value.
 * For TPIC6B595: '1' in storage register turns output ON when latched (OE\ low).
 * No bit inversion needed - direct mapping from bit to valve state.
 * 
 * @note This only shifts data into storage registers, does not latch to outputs.
 *       Latching occurs when DAQ pulses RCK (register clock).
 */
inline void spiShift16(uint16_t value) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer16(value);    // Sends high byte then low byte (MSB-first)
  SPI.endTransaction();
}

/**
 * @brief Send 8-bit data via SPI to shift register chain
 * @param value 8-bit pattern to send
 * 
 * Similar to spiShift16 but for single-byte transfers to smaller chains.
 * Used for switchvalve control which only requires 8 bits.
 * 
 * @note This only shifts data into storage registers, does not latch to outputs.
 *       Latching occurs when DAQ pulses RCK (register clock).
 */
inline void spiShift8(uint8_t value) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer(value);
  SPI.endTransaction();
}

// =====================================================================================
// LOAD REQUEST INTERRUPT SERVICE ROUTINES
// =====================================================================================
//
// These ISRs are triggered when DAQ pulses a LOAD_REQ line (rising edge).
// They read the state bits, look up the corresponding valve pattern,
// preload it into the shift registers via SPI, and assert the READY signal.
//
// The pattern is NOT yet committed to the valve outputs - that happens when
// the DAQ later pulses the RCK line, which triggers the RCK sense ISRs.

/**
 * @brief Load request ISR for left olfactometer
 * 
 * Triggered on rising edge of PIN_OLFACTOMETER_LEFT_LOAD.
 * Reads 3-bit state, looks up valve pattern, preloads via SPI, asserts READY.
 * 
 * @note Only proceeds if no other device currently owns the SPI bus
 * @note Uses digitalReadFast for time-critical state bit reading
 */
void isr_load_olfactometer_left() {
  // Check if SPI bus is available
  if (busOwner != OWNER_NONE) {
    return;  // Another device has a pending preload
  }
  
  // Read 3-bit state from DAQ (S2:S1:S0)
  uint8_t stateIndex = (digitalReadFast(PIN_OLFACTOMETER_LEFT_S2) << 2) |
                       (digitalReadFast(PIN_OLFACTOMETER_LEFT_S1) << 1) |
                       digitalReadFast(PIN_OLFACTOMETER_LEFT_S0);
  
  // Look up and preload valve pattern (mask to ensure valid array index)
  spiShift16(OLFACTOMETER_LEFT_STATES[stateIndex & 0x07]);
  
  // Claim bus ownership and assert ready signal
  readyOlfactometerLeft = true;
  busOwner = OWNER_OLFACTOMETER_LEFT;
  digitalWriteFast(PIN_OLFACTOMETER_LEFT_READY, HIGH);
}

/**
 * @brief Load request ISR for right olfactometer
 * 
 * Triggered on rising edge of PIN_OLFACTOMETER_RIGHT_LOAD.
 * Reads 3-bit state, looks up valve pattern, preloads via SPI, asserts READY.
 */
void isr_load_olfactometer_right() {
  if (busOwner != OWNER_NONE) {
    return;
  }
  
  uint8_t stateIndex = (digitalReadFast(PIN_OLFACTOMETER_RIGHT_S2) << 2) |
                       (digitalReadFast(PIN_OLFACTOMETER_RIGHT_S1) << 1) |
                       digitalReadFast(PIN_OLFACTOMETER_RIGHT_S0);
  
  spiShift16(OLFACTOMETER_RIGHT_STATES[stateIndex & 0x07]);
  
  readyOlfactometerRight = true;
  busOwner = OWNER_OLFACTOMETER_RIGHT;
  digitalWriteFast(PIN_OLFACTOMETER_RIGHT_READY, HIGH);
}

/**
 * @brief Load request ISR for left switchvalve
 * 
 * Triggered on rising edge of PIN_SWITCHVALVE_LEFT_LOAD.
 * Reads 1-bit state, looks up valve pattern, preloads via SPI, asserts READY.
 */
void isr_load_switchvalve_left() {
  if (busOwner != OWNER_NONE) {
    return;
  }
  
  uint8_t stateIndex = digitalReadFast(PIN_SWITCHVALVE_LEFT_S);
  
  spiShift8(SWITCHVALVE_LEFT_STATES[stateIndex & 0x01]);
  
  readySwitchvalveLeft = true;
  busOwner = OWNER_SWITCHVALVE_LEFT;
  digitalWriteFast(PIN_SWITCHVALVE_LEFT_READY, HIGH);
}

/**
 * @brief Load request ISR for right switchvalve
 * 
 * Triggered on rising edge of PIN_SWITCHVALVE_RIGHT_LOAD.
 * Reads 1-bit state, looks up valve pattern, preloads via SPI, asserts READY.
 */
void isr_load_switchvalve_right() {
  if (busOwner != OWNER_NONE) {
    return;
  }
  
  uint8_t stateIndex = digitalReadFast(PIN_SWITCHVALVE_RIGHT_S);
  
  spiShift8(SWITCHVALVE_RIGHT_STATES[stateIndex & 0x01]);
  
  readySwitchvalveRight = true;
  busOwner = OWNER_SWITCHVALVE_RIGHT;
  digitalWriteFast(PIN_SWITCHVALVE_RIGHT_READY, HIGH);
}

// =====================================================================================
// REGISTER CLOCK SENSE INTERRUPT SERVICE ROUTINES
// =====================================================================================
//
// These ISRs are triggered when DAQ pulses an RCK line (rising edge).
// The RCK pulse commits the preloaded pattern from shift register storage
// to the output registers, actually changing the valve states.
//
// The ISRs drop the READY signal and release SPI bus ownership, allowing
// the next device to preload a pattern.

/**
 * @brief RCK sense ISR for left olfactometer
 * 
 * Triggered on rising edge of PIN_RCK_SENSE_OLFACTOMETER_LEFT.
 * Verifies this device owns the bus and is ready, then releases ownership.
 * 
 * @note Only acts if this device currently owns the bus and is in ready state
 * @note DAQ timing ensures RCK pulse occurs only when device is ready
 */
void isr_rck_olfactometer_left() {
  if (busOwner == OWNER_OLFACTOMETER_LEFT && readyOlfactometerLeft) {
    // Pattern has been committed to outputs, release bus
    readyOlfactometerLeft = false;
    busOwner = OWNER_NONE;
    digitalWriteFast(PIN_OLFACTOMETER_LEFT_READY, LOW);
  }
}

/**
 * @brief RCK sense ISR for right olfactometer
 * 
 * Triggered on rising edge of PIN_RCK_SENSE_OLFACTOMETER_RIGHT.
 * Verifies this device owns the bus and is ready, then releases ownership.
 */
void isr_rck_olfactometer_right() {
  if (busOwner == OWNER_OLFACTOMETER_RIGHT && readyOlfactometerRight) {
    readyOlfactometerRight = false;
    busOwner = OWNER_NONE;
    digitalWriteFast(PIN_OLFACTOMETER_RIGHT_READY, LOW);
  }
}

/**
 * @brief RCK sense ISR for left switchvalve
 * 
 * Triggered on rising edge of PIN_RCK_SENSE_SWITCHVALVE_LEFT.
 * Verifies this device owns the bus and is ready, then releases ownership.
 */
void isr_rck_switchvalve_left() {
  if (busOwner == OWNER_SWITCHVALVE_LEFT && readySwitchvalveLeft) {
    readySwitchvalveLeft = false;
    busOwner = OWNER_NONE;
    digitalWriteFast(PIN_SWITCHVALVE_LEFT_READY, LOW);
  }
}

/**
 * @brief RCK sense ISR for right switchvalve
 * 
 * Triggered on rising edge of PIN_RCK_SENSE_SWITCHVALVE_RIGHT.
 * Verifies this device owns the bus and is ready, then releases ownership.
 */
void isr_rck_switchvalve_right() {
  if (busOwner == OWNER_SWITCHVALVE_RIGHT && readySwitchvalveRight) {
    readySwitchvalveRight = false;
    busOwner = OWNER_NONE;
    digitalWriteFast(PIN_SWITCHVALVE_RIGHT_READY, LOW);
  }
}

// =====================================================================================
// ARDUINO SETUP AND MAIN LOOP
// =====================================================================================

/**
 * @brief Arduino setup function - runs once at startup
 * 
 * Initializes all hardware interfaces:
 * 1. SPI communication for shift register control
 * 2. GPIO pins for DAQ interface
 * 3. Interrupt service routines for real-time operation
 * 4. Initial valve states (all OFF)
 * 
 * After setup completes, the system is ready to respond to DAQ commands.
 */
void setup() {
  // ---------------------------------------------------------------------------------
  // SPI Initialization
  // ---------------------------------------------------------------------------------
  
  /**
   * Initialize hardware SPI (SPI0 on Teensy 4.1)
   * - MOSI = Pin 11 (data to shift registers)
   * - SCK  = Pin 13 (clock to shift registers)
   * - Keep hardware CS as output per SPI library convention (not used)
   */
  pinMode(10, OUTPUT);  // Hardware CS pin (unused but required as output)
  SPI.begin();          // Initialize SPI with default pins

  
  // ---------------------------------------------------------------------------------
  // READY Signal Pin Configuration
  // ---------------------------------------------------------------------------------
  
  /**
   * Configure READY output pins (to DAQ)
   * These signals indicate when Teensy has preloaded a pattern and is ready
   * for the DAQ to pulse RCK to commit the pattern to valve outputs.
   */
  pinMode(PIN_OLFACTOMETER_LEFT_READY, OUTPUT);
  pinMode(PIN_OLFACTOMETER_RIGHT_READY, OUTPUT);
  pinMode(PIN_SWITCHVALVE_LEFT_READY, OUTPUT);
  pinMode(PIN_SWITCHVALVE_RIGHT_READY, OUTPUT);
  
  // Initialize all READY signals to LOW (not ready)
  digitalWriteFast(PIN_OLFACTOMETER_LEFT_READY, LOW);
  digitalWriteFast(PIN_OLFACTOMETER_RIGHT_READY, LOW);
  digitalWriteFast(PIN_SWITCHVALVE_LEFT_READY, LOW);
  digitalWriteFast(PIN_SWITCHVALVE_RIGHT_READY, LOW);

  
  // ---------------------------------------------------------------------------------
  // State Input Pin Configuration
  // ---------------------------------------------------------------------------------
  
  /**
   * Configure state input pins (from DAQ)
   * These pins carry the requested valve state before each LOAD_REQ
   */
  
  // Left olfactometer: 3-bit state (8 possible states)
  pinMode(PIN_OLFACTOMETER_LEFT_S0, INPUT);
  pinMode(PIN_OLFACTOMETER_LEFT_S1, INPUT);
  pinMode(PIN_OLFACTOMETER_LEFT_S2, INPUT);
  
  // Right olfactometer: 3-bit state (8 possible states)
d  pinMode(PIN_OLFACTOMETER_RIGHT_S0, INPUT);
  pinMode(PIN_OLFACTOMETER_RIGHT_S1, INPUT);
  pinMode(PIN_OLFACTOMETER_RIGHT_S2, INPUT);
  
  // Switchvalves: 1-bit state each (2 possible states)
  pinMode(PIN_SWITCHVALVE_LEFT_S, INPUT);
  pinMode(PIN_SWITCHVALVE_RIGHT_S, INPUT);

  
  // ---------------------------------------------------------------------------------
  // Load Request Input Pin Configuration
  // ---------------------------------------------------------------------------------
  
  /**
   * Configure LOAD_REQ input pins (from DAQ)
   * Rising edges on these pins trigger the load ISRs
   */
  pinMode(PIN_OLFACTOMETER_LEFT_LOAD, INPUT);
  pinMode(PIN_OLFACTOMETER_RIGHT_LOAD, INPUT);
  pinMode(PIN_SWITCHVALVE_LEFT_LOAD, INPUT);
  pinMode(PIN_SWITCHVALVE_RIGHT_LOAD, INPUT);

  // RCK sense inputs (from DAQ’s RCK lines)
  pinMode(PIN_RCK_SENSE_OLFACTOMETER_LEFT, INPUT);
  pinMode(PIN_RCK_SENSE_OLFACTOMETER_RIGHT, INPUT);
  pinMode(PIN_RCK_SENSE_SWITCHVALVE_LEFT, INPUT);
  pinMode(PIN_RCK_SENSE_SWITCHVALVE_RIGHT, INPUT);

  
  // ---------------------------------------------------------------------------------
  // Interrupt Service Routine Attachment
  // ---------------------------------------------------------------------------------
  
  /**
   * Attach load request interrupts (triggered on rising edge)
   * These ISRs preload valve patterns and assert READY signals
   */
  attachInterrupt(digitalPinToInterrupt(PIN_OLFACTOMETER_LEFT_LOAD), 
                  isr_load_olfactometer_left, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_OLFACTOMETER_RIGHT_LOAD), 
                  isr_load_olfactometer_right, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_SWITCHVALVE_LEFT_LOAD), 
                  isr_load_switchvalve_left, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_SWITCHVALVE_RIGHT_LOAD), 
                  isr_load_switchvalve_right, RISING);
  
  /**
   * Attach RCK sense interrupts (triggered on rising edge)
   * These ISRs detect pattern commits and release SPI bus ownership
   */
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLFACTOMETER_LEFT), 
                  isr_rck_olfactometer_left, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLFACTOMETER_RIGHT), 
                  isr_rck_olfactometer_right, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SWITCHVALVE_LEFT), 
                  isr_rck_switchvalve_left, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SWITCHVALVE_RIGHT), 
                  isr_rck_switchvalve_right, RISING);

  
  // ---------------------------------------------------------------------------------
  // Initial Valve State Setup
  // ---------------------------------------------------------------------------------
  
  /**
   * Preload all devices to OFF state
   * This ensures a known initial state. The DAQ can latch these patterns
   * once the system is fully armed and operational.
   * 
   * Note: These patterns are preloaded but not committed until DAQ pulses RCK
   */
  spiShift16(OLFACTOMETER_LEFT_STATES[ST_OFF]);   // Left olfactometer OFF
  spiShift16(OLFACTOMETER_RIGHT_STATES[ST_OFF]);  // Right olfactometer OFF
  spiShift8(SWITCHVALVE_LEFT_STATES[0]);          // Left switchvalve CLEAN
  spiShift8(SWITCHVALVE_RIGHT_STATES[0]);         // Right switchvalve CLEAN
  
  /**
   * Setup complete - system is now ready for DAQ operation
   * No READY signals are asserted yet (not waiting for any commits)
   * DAQ may initiate the first LOAD_REQ at any time
   */
}

/**
 * @brief Arduino main loop - runs continuously
 * 
 * The main loop is intentionally empty because all real-time valve control
 * is handled by interrupt service routines. This design ensures:
 * - Deterministic response times to DAQ commands
 * - Minimal jitter in valve timing
 * - No blocking operations in the critical path
 * 
 * The system operates entirely in interrupt-driven mode:
 * 1. LOAD_REQ interrupts preload valve patterns
 * 2. RCK sense interrupts detect pattern commits
 * 3. SPI bus arbitration prevents conflicts
 */
void loop() {
  // Intentionally empty - all work is interrupt-driven for real-time performance
  // 
  // Future enhancements could add:
  // - Serial communication for debugging/status
  // - Watchdog timer management
  // - Error detection and reporting
  // - Performance monitoring
}
