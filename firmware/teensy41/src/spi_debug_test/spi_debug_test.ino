/**
 * @file spi_debug_test.ino
 * @brief SPI Debug Test for MultiBiOS - Oscilloscope-friendly patterns
 * @version 1.0
 * 
 * This firmware generates simple, predictable SPI patterns for debugging
 * the serial output chain with an oscilloscope. No interrupts or complex
 * logic - just repeating test patterns.
 * 
 * Test Patterns:
 * 1. All zeros (0x0000, 0x0000, 0x00, 0x00)
 * 2. All ones (0xFFFF, 0xFFFF, 0xFF, 0xFF) 
 * 3. Alternating bits (0x5555, 0xAAAA, 0x55, 0xAA)
 * 4. Walking ones (single bit shifts through all positions)
 * 5. Sequential counting patterns
 * 
 * Wiring for SPI output test:
 *   MOSI (Pin 11) -> Connect to oscilloscope probe 1
 *   SCK  (Pin 13) -> Connect to oscilloscope probe 2  
 *   Pin 10 (SS)   -> Set as output (SPI requirement)
 *   
 * Expected 48-bit frame format:
 *   [16-bit OLFA_L][16-bit OLFA_R][8-bit SW_L][8-bit SW_R]
 */

#include <Arduino.h>
#include <SPI.h>

// -------------------- Pins --------------------
constexpr int PIN_MOSI = 11;
constexpr int PIN_SCK  = 13;
constexpr int PIN_SS   = 10;  // Required for SPI

// -------------------- SPI Config --------------------
uint32_t SPI_HZ = 1'000'000;  // Start slower for easier scope viewing (non-const for runtime changes)

// -------------------- Test Mode Selection --------------------
enum TestMode {
  MODE_ALL_ZEROS = 0,
  MODE_ALL_ONES = 1,
  MODE_ALTERNATING = 2,
  MODE_WALKING_ONES = 3,
  MODE_COUNTING = 4,
  MODE_CUSTOM_PATTERNS = 5,
  NUM_MODES = 6
};

volatile TestMode currentMode = MODE_ALL_ZEROS;
volatile uint32_t patternCounter = 0;

// -------------------- Debug LED --------------------
constexpr int LED_PIN = LED_BUILTIN;  // Use built-in LED

// -------------------- SPI Helper Functions --------------------
inline void spiShift16(uint16_t value) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer16(value);
  SPI.endTransaction();
}

inline void spiShift8(uint8_t value) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer(value);
  SPI.endTransaction();
}

// Send complete 48-bit frame
void sendTestFrame(uint16_t olfa_l, uint16_t olfa_r, uint8_t sw_l, uint8_t sw_r) {
  digitalWrite(LED_PIN, HIGH);  // Visual indicator of transmission
  
  spiShift16(olfa_l);
  spiShift16(olfa_r);
  spiShift8(sw_l);
  spiShift8(sw_r);
  
  digitalWrite(LED_PIN, LOW);
}

// -------------------- Test Pattern Generators --------------------
void generateAllZeros() {
  sendTestFrame(0x0000, 0x0000, 0x00, 0x00);
  Serial.println("Pattern: ALL ZEROS - 0x0000 0x0000 0x00 0x00");
}

void generateAllOnes() {
  sendTestFrame(0xFFFF, 0xFFFF, 0xFF, 0xFF);
  Serial.println("Pattern: ALL ONES - 0xFFFF 0xFFFF 0xFF 0xFF");
}

void generateAlternating() {
  // Alternate between 0x5555/0xAAAA and 0xAAAA/0x5555 each call
  if (patternCounter % 2 == 0) {
    sendTestFrame(0x5555, 0xAAAA, 0x55, 0xAA);
    Serial.println("Pattern: ALTERNATING A - 0x5555 0xAAAA 0x55 0xAA");
  } else {
    sendTestFrame(0xAAAA, 0x5555, 0xAA, 0x55);
    Serial.println("Pattern: ALTERNATING B - 0xAAAA 0x5555 0xAA 0x55");
  }
}

void generateWalkingOnes() {
  // Walk a single '1' bit through all 48 positions
  uint8_t bitPos = patternCounter % 48;
  
  uint16_t olfa_l = 0, olfa_r = 0;
  uint8_t sw_l = 0, sw_r = 0;
  
  if (bitPos < 16) {
    // Bit in OLFA_L (positions 0-15)
    olfa_l = 1 << bitPos;
  } else if (bitPos < 32) {
    // Bit in OLFA_R (positions 16-31)
    olfa_r = 1 << (bitPos - 16);
  } else if (bitPos < 40) {
    // Bit in SW_L (positions 32-39)
    sw_l = 1 << (bitPos - 32);
  } else {
    // Bit in SW_R (positions 40-47)
    sw_r = 1 << (bitPos - 40);
  }
  
  sendTestFrame(olfa_l, olfa_r, sw_l, sw_r);
  Serial.print("Pattern: WALKING ONES bit ");
  Serial.print(bitPos);
  Serial.print(" - 0x");
  Serial.print(olfa_l, HEX);
  Serial.print(" 0x");
  Serial.print(olfa_r, HEX);
  Serial.print(" 0x");
  Serial.print(sw_l, HEX);
  Serial.print(" 0x");
  Serial.println(sw_r, HEX);
}

void generateCounting() {
  // Simple incrementing counter across all fields
  uint16_t count16 = patternCounter & 0xFFFF;
  uint8_t count8 = patternCounter & 0xFF;
  
  sendTestFrame(count16, ~count16, count8, ~count8);
  Serial.print("Pattern: COUNTING ");
  Serial.print(patternCounter);
  Serial.print(" - 0x");
  Serial.print(count16, HEX);
  Serial.print(" 0x");
  Serial.print(~count16 & 0xFFFF, HEX);
  Serial.print(" 0x");
  Serial.print(count8, HEX);
  Serial.print(" 0x");
  Serial.println((~count8) & 0xFF, HEX);
}

void generateCustomPatterns() {
  // Custom test patterns for specific debugging
  switch (patternCounter % 4) {
    case 0:
      // Test pattern: each nibble different
      sendTestFrame(0x1234, 0x5678, 0x9A, 0xBC);
      Serial.println("Pattern: CUSTOM 1 - 0x1234 0x5678 0x9A 0xBC");
      break;
    case 1:
      // Test pattern: powers of 2
      sendTestFrame(0x0001, 0x0002, 0x04, 0x08);
      Serial.println("Pattern: CUSTOM 2 - 0x0001 0x0002 0x04 0x08");
      break;
    case 2:
      // Test pattern: known valve states
      sendTestFrame(0x0003, 0x000C, 0x03, 0x00);  // Air valves
      Serial.println("Pattern: CUSTOM 3 - 0x0003 0x000C 0x03 0x00");
      break;
    case 3:
      // Test pattern: flush state
      sendTestFrame(0x0FFF, 0x0FFF, 0xFF, 0xFF);
      Serial.println("Pattern: CUSTOM 4 - 0x0FFF 0x0FFF 0xFF 0xFF");
      break;
  }
}

// -------------------- Mode Control --------------------
void switchToNextMode() {
  currentMode = static_cast<TestMode>((currentMode + 1) % NUM_MODES);
  patternCounter = 0;  // Reset pattern counter for new mode
  
  Serial.println("\n========================================");
  switch (currentMode) {
    case MODE_ALL_ZEROS:
      Serial.println("SWITCHED TO MODE: ALL ZEROS");
      Serial.println("Scope: Look for constant LOW on MOSI");
      break;
    case MODE_ALL_ONES:
      Serial.println("SWITCHED TO MODE: ALL ONES");
      Serial.println("Scope: Look for constant HIGH on MOSI");
      break;
    case MODE_ALTERNATING:
      Serial.println("SWITCHED TO MODE: ALTERNATING BITS");
      Serial.println("Scope: Look for regular HIGH/LOW pattern");
      break;
    case MODE_WALKING_ONES:
      Serial.println("SWITCHED TO MODE: WALKING ONES");
      Serial.println("Scope: Look for single bit walking through frame");
      break;
    case MODE_COUNTING:
      Serial.println("SWITCHED TO MODE: COUNTING");
      Serial.println("Scope: Look for incrementing bit patterns");
      break;
    case MODE_CUSTOM_PATTERNS:
      Serial.println("SWITCHED TO MODE: CUSTOM PATTERNS");
      Serial.println("Scope: Look for specific test patterns");
      break;
    case NUM_MODES:
      // This should never happen due to modulo operation, but included for completeness
      Serial.println("ERROR: Invalid mode");
      currentMode = MODE_ALL_ZEROS;
      break;
  }
  Serial.println("========================================\n");
}

// -------------------- Setup --------------------
void setup() {
  // Serial for debug output
  Serial.begin(115200);
  delay(2000);  // Wait for serial monitor
  
  Serial.println("=== MultiBiOS SPI Debug Test ===");
  Serial.println("Version 1.0");
  Serial.println();
  
  // Basic pin tests BEFORE SPI initialization
  Serial.println("=== BASIC PIN TESTS ===");
  
  // Test LED first
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.println("Testing built-in LED...");
  for (int i = 0; i < 5; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(200);
    digitalWrite(LED_BUILTIN, LOW);
    delay(200);
  }
  Serial.println("LED test complete");
  
  // Test manual pin control BEFORE SPI
  Serial.println("Testing manual pin control...");
  pinMode(PIN_MOSI, OUTPUT);
  pinMode(PIN_SCK, OUTPUT);
  pinMode(PIN_SS, OUTPUT);
  
  // Manual toggle test
  Serial.println("Manual MOSI toggle test (should see on scope)...");
  for (int i = 0; i < 20; i++) {
    digitalWrite(PIN_MOSI, HIGH);
    delayMicroseconds(100);
    digitalWrite(PIN_MOSI, LOW);
    delayMicroseconds(100);
  }
  
  Serial.println("Manual SCK toggle test (should see on scope)...");
  for (int i = 0; i < 20; i++) {
    digitalWrite(PIN_SCK, HIGH);
    delayMicroseconds(100);
    digitalWrite(PIN_SCK, LOW);
    delayMicroseconds(100);
  }
  
  Serial.println("Manual pin tests complete");
  Serial.println("*** CHECK SCOPE NOW - You should have seen signals ***");
  Serial.println();
  
  // Now initialize SPI
  Serial.println("=== SPI INITIALIZATION ===");
  digitalWrite(PIN_SS, HIGH);  // Deassert SS before SPI.begin()
  
  // Initialize SPI
  SPI.begin();
  Serial.print("SPI initialized at ");
  Serial.print(SPI_HZ);
  Serial.println(" Hz");
  
  // Test SPI with simple pattern
  Serial.println("Testing SPI transfer...");
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  SPI.transfer(0xAA);  // Simple test byte
  SPI.endTransaction();
  Serial.println("SPI test transfer complete");
  
  // Test our SPI helper functions
  Serial.println("Testing SPI helper functions...");
  sendTestFrame(0xA5A5, 0x5A5A, 0xA5, 0x5A);
  Serial.println("SPI helper test complete");
  
  Serial.println("\nOscilloscope Setup Instructions:");
  Serial.println("- Connect MOSI (Pin 11) to Channel 1");
  Serial.println("- Connect SCK (Pin 13) to Channel 2");
  Serial.println("- Set timebase to 10us/div to start");
  Serial.println("- Trigger on Channel 2 (SCK) rising edge");
  Serial.println("- Look for 48 bits per frame (16+16+8+8)");
  Serial.println();
  
  Serial.println("Control Commands:");
  Serial.println("- Send 'n' to switch to next test mode");
  Serial.println("- Send 's' to change SPI speed");
  Serial.println("- Send 'r' to reset pattern counter");
  Serial.println("- Send 't' to run manual pin toggle test");
  Serial.println("- Send 'p' to test individual SPI transfers");
  Serial.println();
  
  // Start with first mode
  switchToNextMode();
}

// -------------------- Main Loop --------------------
void loop() {
  // Check for serial commands
  if (Serial.available()) {
    char cmd = Serial.read();
    switch (cmd) {
      case 'n':
      case 'N':
        switchToNextMode();
        break;
      case 's':
      case 'S':
        // Cycle through different SPI speeds
        Serial.println("SPI Speed options - enter number:");
        Serial.println("1: 100 kHz (very slow)");
        Serial.println("2: 1 MHz (slow)");
        Serial.println("3: 5 MHz (medium)");
        Serial.println("4: 10 MHz (fast)");
        break;
      case '1':
        SPI_HZ = 100000;
        Serial.println("SPI speed set to 100 kHz");
        break;
      case '2':
        SPI_HZ = 1000000;
        Serial.println("SPI speed set to 1 MHz");
        break;
      case '3':
        SPI_HZ = 5000000;
        Serial.println("SPI speed set to 5 MHz");
        break;
      case '4':
        SPI_HZ = 10000000;
        Serial.println("SPI speed set to 10 MHz");
        break;
      case 'r':
      case 'R':
        patternCounter = 0;
        Serial.println("Pattern counter reset");
        break;
      case 't':
      case 'T':
        Serial.println("Running manual pin toggle test...");
        for (int i = 0; i < 50; i++) {
          digitalWrite(PIN_MOSI, HIGH);
          digitalWrite(PIN_SCK, HIGH);
          delayMicroseconds(50);
          digitalWrite(PIN_MOSI, LOW);
          digitalWrite(PIN_SCK, LOW);
          delayMicroseconds(50);
        }
        Serial.println("Manual toggle test complete - check scope!");
        break;
      case 'p':
      case 'P':
        Serial.println("Testing individual SPI transfers...");
        SPI.beginTransaction(SPISettings(100000, MSBFIRST, SPI_MODE0)); // Very slow
        for (int i = 0; i < 10; i++) {
          SPI.transfer(0x55); // Alternating bits
          delay(10);
        }
        SPI.endTransaction();
        Serial.println("Individual SPI test complete - check scope!");
        break;
    }
  }
  
  // Generate test pattern based on current mode
  switch (currentMode) {
    case MODE_ALL_ZEROS:
      generateAllZeros();
      break;
    case MODE_ALL_ONES:
      generateAllOnes();
      break;
    case MODE_ALTERNATING:
      generateAlternating();
      break;
    case MODE_WALKING_ONES:
      generateWalkingOnes();
      break;
    case MODE_COUNTING:
      generateCounting();
      break;
    case MODE_CUSTOM_PATTERNS:
      generateCustomPatterns();
      break;
    case NUM_MODES:
      // This should never happen, but included for completeness
      Serial.println("ERROR: Invalid mode, resetting to ALL ZEROS");
      currentMode = MODE_ALL_ZEROS;
      generateAllZeros();
      break;
  }
  
  patternCounter++;
  
  // Delay between patterns (adjustable)
  delay(500);  // 500ms between transmissions for easy scope viewing
}