#include <Arduino.h>
#include <SPI.h>

// ---------- SPI ----------
constexpr uint32_t SPI_HZ = 1'000'000;
constexpr int PIN_CS = 10;  // Teensy SPI requirement

// ---------- Chain ----------
constexpr int NUM_TPICS = 6;          // 6 bytes = 48 bits total

// Your physical order (nearest → farthest): OLF1_LO, OLF1_HI, OLF2_LO, OLF2_HI, SV2, SV1
// We’ll build 6 logical bytes, then map to this send order:
// out[5] (sent first) → farthest (SV1), out[0] (sent last) → nearest (OLF1_LO).
enum FrameSlot : uint8_t { SLOT_OLF1_HI=0, SLOT_OLF1_LO, SLOT_OLF2_HI, SLOT_OLF2_LO, SLOT_SV1, SLOT_SV2 };
uint8_t FRAME_SEND_ORDER[6] = {
  SLOT_OLF1_LO,  // out[0] → nearest (OLF1_LO)
  SLOT_OLF1_HI,  // out[1] → OLF1_HI
  SLOT_OLF2_LO,  // out[2] → OLF2_LO
  SLOT_OLF2_HI,  // out[3] → OLF2_HI
  SLOT_SV2,      // out[4] → SV2
  SLOT_SV1       // out[5] → farthest (SV1)
};

// ---------- Allowed bit list ----------
constexpr uint8_t kAllowedBits[] = {
  // 0..11
  0,1,2,3,4,5,6,7,8,9,10,11,
  // 16..27
  16,17,18,19,20,21,22,23,24,25,26,27,
  // singles
  32,33,40,41
};
constexpr size_t kAllowedCount = sizeof(kAllowedBits) / sizeof(kAllowedBits[0]);

// ---------- Timing ----------
elapsedMillis phaseTimer;
bool onPhase = true;               // ON then OFF
size_t allowedIdx = 0;             // index into kAllowedBits[]
uint16_t bitIndex = kAllowedBits[0];

// ---------- Helpers ----------
static inline void spi_send_48(const uint8_t out[6]) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  // Send far → near: out[5] first ... out[0] last
  for (int i = 5; i >= 0; --i) SPI.transfer(out[i]);
  SPI.endTransaction();
}

static inline void send_all_zero() {
  uint8_t out[6] = {0,0,0,0,0,0};
  spi_send_48(out);
  Serial.println("OFF gap: all zeros sent");
}

static inline void send_one_hot(uint16_t idx) {
  // Build logical slots
  uint8_t slots[6] = {0,0,0,0,0,0};
  uint8_t byteIdx = idx / 8;     // 0..5
  uint8_t bitInByte = idx % 8;   // 0..7 (LSB=Q0)

  // byteIdx mapping: 0=OLF1_LO, 1=OLF1_HI, 2=OLF2_LO, 3=OLF2_HI, 4=SV2, 5=SV1
  const uint8_t byteIdx_to_slot[6] = {
    SLOT_OLF1_LO, SLOT_OLF1_HI, SLOT_OLF2_LO, SLOT_OLF2_HI, SLOT_SV2, SLOT_SV1
  };

  slots[ byteIdx_to_slot[byteIdx] ] = (uint8_t)(1u << bitInByte);

  // Map slots → send buffer according to FRAME_SEND_ORDER
  uint8_t out[6];
  for (int i = 0; i < 6; ++i) out[i] = slots[ FRAME_SEND_ORDER[i] ];

  // Send it
  spi_send_48(out);

  // Print for visibility
  Serial.print("ON bit ");
  Serial.print(idx);
  Serial.print("  (byte=");
  Serial.print(byteIdx);
  Serial.print(", bit=");
  Serial.print(bitInByte);
  Serial.print(")  Data: ");
  for (int i = 5; i >= 0; --i) {
    if (out[i] < 0x10) Serial.print('0');
    Serial.print(out[i], HEX);
    Serial.print(' ');
  }
  Serial.println();
}

static inline void advance_to_next_allowed() {
  allowedIdx = (allowedIdx + 1) % kAllowedCount;
  bitIndex = kAllowedBits[allowedIdx];
}

// ---------- Setup / Loop ----------
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("=== TPIC6B595 debug: walking only allowed bits ===");
  Serial.println("Allowed sets: 0–11, 16–27, 32,33,40,41 | 500 ms ON, 500 ms OFF");

  pinMode(PIN_CS, OUTPUT);   // required for SPI on Teensy
  SPI.begin();

  // Start with the first ON frame
  send_one_hot(bitIndex);
  phaseTimer = 0;
  onPhase = true;
}

void loop() {
  // 500 ms on, then 500 ms off, then advance to next allowed bit
  if (onPhase && phaseTimer >= 500) {
    onPhase = false;
    phaseTimer = 0;
    send_all_zero();
  } else if (!onPhase && phaseTimer >= 500) {
    onPhase = true;
    phaseTimer = 0;
    advance_to_next_allowed();
    send_one_hot(bitIndex);
  }
}
