#include <Arduino.h>
#include <SPI.h>

// ---- SPI ----
constexpr int PIN_CS  = 10;              // dummy CS (must be OUTPUT on Teensy)
constexpr uint32_t SPI_HZ = 1000000;     // slow enough to see on a scope

// ---- TPIC chain ----
constexpr int NUM_TPICS = 6;             // 6 x TPIC6B595 = 48 outputs total
uint8_t frame[NUM_TPICS];                // 6 bytes

// walking bit state
uint32_t patternIndex = 0;
elapsedMillis tick;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("=== TPIC6B595 6x chain test (48 bits) ===");
  Serial.println("Sends 6 bytes (48 bits). Latch with your DAQ RCK ↑ every 1s.");

  pinMode(PIN_CS, OUTPUT);               // required on Teensy
  SPI.begin();
  memset(frame, 0x00, sizeof(frame));
}

void loop() {
  if (tick >= 1000) {                    // new pattern every 1s
    tick = 0;
    patternIndex++;

    // ---- build 48-bit walking '1' ----
    memset(frame, 0x00, sizeof(frame));
    const int totalBits = NUM_TPICS * 8;           // 48
    int bitPos = patternIndex % totalBits;         // 0..47
    frame[bitPos / 8] = (uint8_t)(1u << (bitPos % 8));

    // ---- send 6 bytes (MSB first per byte, last byte ends up in nearest TPIC) ----
    // For 595-style chains: the FIRST byte shifted goes to the FARTHEST device.
    // Therefore send from highest index down to 0 so frame[0] goes to the NEAREST TPIC.
    SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
    for (int i = NUM_TPICS - 1; i >= 0; --i) {
      SPI.transfer(frame[i]);
    }
    SPI.endTransaction();
    // Your DAQ should now pulse RCK ↑ to latch these 48 bits

    // ---- serial print ----
    Serial.print("Pattern #");
    Serial.print(patternIndex);
    Serial.print("  Bit ");
    Serial.print(bitPos);
    Serial.print(" HIGH  |  Data: ");
    for (int i = NUM_TPICS - 1; i >= 0; --i) {     // print in shift order
      if (frame[i] < 0x10) Serial.print('0');
      Serial.print(frame[i], HEX);
      Serial.print(' ');
    }
    Serial.println();
  }
}
