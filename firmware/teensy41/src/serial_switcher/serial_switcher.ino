#include <Arduino.h>
#include <SPI.h>

// ---------- SPI ----------
constexpr uint32_t SPI_HZ = 1'000'000;
constexpr int PIN_CS = 10;  // Teensy SPI requirement

// ---------- Chain ----------
constexpr int NUM_TPICS = 6; // 6 bytes = 48 bits total

// Physical order (nearest → farthest): OLF1_LO, OLF1_HI, OLF2_LO, OLF2_HI, SV2, SV1
// We build 6 logical bytes, then map to this send order:
// out[5] (sent first) → farthest (SV1), out[0] (last) → nearest (OLF1_LO).
enum FrameSlot : uint8_t { SLOT_OLF1_HI=0, SLOT_OLF1_LO, SLOT_OLF2_HI, SLOT_OLF2_LO, SLOT_SV1, SLOT_SV2 };
uint8_t FRAME_SEND_ORDER[6] = {
  SLOT_OLF1_LO,  // out[0] → nearest (OLF1_LO)
  SLOT_OLF1_HI,  // out[1] → OLF1_HI
  SLOT_OLF2_LO,  // out[2] → OLF2_LO
  SLOT_OLF2_HI,  // out[3] → OLF2_HI
  SLOT_SV2,      // out[4] → SV2
  SLOT_SV1       // out[5] → farthest (SV1)
};

// ---------- Staged state (not sent until SEND) ----------
uint16_t OLF1_bits = 0;  // 12 LSBs used: OLF1_0..OLF1_11
uint16_t OLF2_bits = 0;  // 12 LSBs used: OLF2_0..OLF2_11
uint8_t  SV1_bits  = 0;  // 2  LSBs used: SV1_0..SV1_1
uint8_t  SV2_bits  = 0;  // 2  LSBs used: SV2_0..SV2_1

// ---------- Status print cadence ----------
elapsedMillis statusTimer;
constexpr uint32_t STATUS_MS = 1000; // print once per second

// ---------- Helpers ----------
static inline void spi_send_48(const uint8_t out[6]) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  // Send far → near: out[5] first ... out[0] last
  for (int i = 5; i >= 0; --i) SPI.transfer(out[i]);
  SPI.endTransaction();
}

static inline void build_send_buffer(uint8_t out[6]) {
  uint8_t slots[6] = {0,0,0,0,0,0};

  uint8_t OLF1_LO = (uint8_t)(OLF1_bits & 0xFF);
  uint8_t OLF1_HI = (uint8_t)((OLF1_bits >> 8) & 0x0F); // 4 LSBs used
  uint8_t OLF2_LO = (uint8_t)(OLF2_bits & 0xFF);
  uint8_t OLF2_HI = (uint8_t)((OLF2_bits >> 8) & 0x0F); // 4 LSBs used

  slots[SLOT_OLF1_LO] = OLF1_LO;
  slots[SLOT_OLF1_HI] = OLF1_HI;
  slots[SLOT_OLF2_LO] = OLF2_LO;
  slots[SLOT_OLF2_HI] = OLF2_HI;
  slots[SLOT_SV1]     = (uint8_t)(SV1_bits & 0x03);
  slots[SLOT_SV2]     = (uint8_t)(SV2_bits & 0x03);

  for (int i = 0; i < 6; ++i) out[i] = slots[ FRAME_SEND_ORDER[i] ];
}

static inline void do_send() {
  uint8_t out[6];
  build_send_buffer(out);
  spi_send_48(out);

  Serial.print("SEND  Data: ");
  for (int i = 5; i >= 0; --i) {
    if (out[i] < 0x10) Serial.print('0');
    Serial.print(out[i], HEX);
    Serial.print(' ');
  }
  Serial.println();
}

static inline void print_states() {
  // 12 bits OLF1, space, 12 bits OLF2, space, 2 bits SV1, space, 2 bits SV2
  char line[64]; int p = 0;

  for (int b = 11; b >= 0; --b) line[p++] = ( (OLF1_bits >> b) & 1 ) ? '1' : '0';
  line[p++] = ' ';
  for (int b = 11; b >= 0; --b) line[p++] = ( (OLF2_bits >> b) & 1 ) ? '1' : '0';
  line[p++] = ' ';
  for (int b = 1; b >= 0; --b) line[p++] = ( (SV1_bits  >> b) & 1 ) ? '1' : '0';
  line[p++] = ' ';
  for (int b = 1; b >= 0; --b) line[p++] = ( (SV2_bits  >> b) & 1 ) ? '1' : '0';

  line[p] = 0;
  Serial.println(line);
}

static inline void clear_states() {
  OLF1_bits = 0;
  OLF2_bits = 0;
  SV1_bits  = 0;
  SV2_bits  = 0;
}

// ---- bitfield mutators used by aliases + manual commands ----
static inline bool set_one(const String& key, bool value, bool toggle=false) {
  auto apply = [&](uint16_t &field, int idx, int maxBits) -> bool {
    if (idx < 0 || idx >= maxBits) return false;
    uint16_t mask = (1u << idx);
    if (toggle) field ^= mask;
    else if (value) field |= mask;
    else field &= ~mask;
    return true;
  };
  auto apply2 = [&](uint8_t &field, int idx, int maxBits) -> bool {
    if (idx < 0 || idx >= maxBits) return false;
    uint8_t mask = (1u << idx);
    if (toggle) field ^= mask;
    else if (value) field |= mask;
    else field &= ~mask;
    return true;
  };

  String s = key; s.toUpperCase();
  int us = s.indexOf('_');
  if (us < 0) return false;
  String name = s.substring(0, us); // OLF1 / OLF2 / SV1 / SV2
  int idx = s.substring(us+1).toInt();

  if (name == "OLF1") return apply(OLF1_bits, idx, 12);
  if (name == "OLF2") return apply(OLF2_bits, idx, 12);
  if (name == "SV1")  return apply2(SV1_bits,  idx, 2);
  if (name == "SV2")  return apply2(SV2_bits,  idx, 2);
  return false;
}

// ---- alias helpers ----
static inline void olf_set_pair(uint16_t &field, uint8_t firstBit /*0,2,4,6,8,10*/) {
  // OFF all 12, then ON firstBit & firstBit+1
  field = 0;
  if (firstBit < 12) field |= (1u << firstBit);
  if (firstBit + 1 < 12) field |= (1u << (firstBit + 1));
}

static inline bool apply_air_or_odor(const String& cmd, const String& target) {
  // AIR -> pair (0,1); ODORn -> pair (2n, 2n+1), n in [1..5]
  uint8_t first = 0;
  if (cmd == "AIR") first = 0;
  else if (cmd.startsWith("ODOR")) {
    int n = cmd.substring(4).toInt();  // e.g., ODOR3 -> 3
    if (n < 1 || n > 5) return false;
    first = (uint8_t)(2 * n);
  } else {
    return false;
  }

  String t = target; t.toUpperCase();
  if (t == "ALL") {
    olf_set_pair(OLF1_bits, first);
    olf_set_pair(OLF2_bits, first);
  } else if (t == "OLF1") {
    olf_set_pair(OLF1_bits, first);
  } else if (t == "OLF2") {
    olf_set_pair(OLF2_bits, first);
  } else {
    return false;
  }
  return true;
}

static inline bool apply_clean_or_odor_switch(const String& cmd, const String& which) {
  bool odor = (cmd == "ODOR");
  String w = which; w.toUpperCase();
  if (w == "SV1") {
    SV1_bits = odor ? 0b11 : 0b00;
  } else if (w == "SV2") {
    SV2_bits = odor ? 0b11 : 0b00;
  } else if (w == "ALL") { // (optional convenience)
    SV1_bits = SV2_bits = odor ? 0b11 : 0b00;
  } else {
    return false;
  }
  return true;
}

// ---------- Command parsing (with Option A signed/unsigned fix) ----------
static inline void handle_command_line(String line) {
  line.trim();
  if (line.length() == 0) return;

  const int MAX_TOK = 32;
  String tok[MAX_TOK];
  int ntok = 0;

  int start = 0;
  const int lineLen = (int)line.length();  // Option A: normalize to signed int

  while (start < lineLen && ntok < MAX_TOK) {
    int sp = line.indexOf(' ', start);
    if (sp < 0) sp = lineLen;
    if (sp > start) tok[ntok++] = line.substring(start, sp);
    start = sp + 1;
  }
  if (ntok == 0) return;

  String cmd = tok[0]; cmd.toUpperCase();

  // ---- HELP ----
  if (cmd == "HELP") {
    Serial.println(F(
      "Commands:\n"
      "  ON <name> [name...]      e.g. ON OLF1_0 OLF2_6 SV1_1\n"
      "  OFF <name> [name...]     e.g. OFF OLF1_3 SV2_0\n"
      "  TOGGLE <name> [name...]  e.g. TOGGLE OLF1_0 SV1_1\n"
      "  CLEAR | CLEAN            set all staged bits to 0 (no send)\n"
      "  RESET                    CLEAR then SEND\n"
      "  SEND                     transmit current staged state to TPICs\n"
      "  PRINT                    print staged state once\n"
      "Aliases:\n"
      "  AIR <OLF1|OLF2|ALL>\n"
      "  ODOR1..ODOR5 <OLF1|OLF2|ALL>\n"
      "  CLEAN <SV1|SV2>\n"
      "  ODOR  <SV1|SV2>\n"
      "Names: OLF1_0..OLF1_11, OLF2_0..OLF2_11, SV1_0..SV1_1, SV2_0..SV2_1"
    ));
    return;
  }

  // ---- PRINT / CLEAR / RESET / SEND ----
  if (cmd == "PRINT") { print_states(); return; }

  if (cmd == "CLEAR" || cmd == "CLEAN") {
    clear_states();
    Serial.println("CLEARED");
    return;
  }

  if (cmd == "RESET") {
    clear_states();
    do_send();
    Serial.println("RESET OK");
    return;
  }

  if (cmd == "SEND") { do_send(); return; }

  // ---- Aliases: AIR / ODORn ----
  if (cmd == "AIR" || cmd.startsWith("ODOR")) {
    if (ntok < 2) { Serial.println("ERR: Missing target (use OLF1 | OLF2 | ALL)"); return; }
    if (!apply_air_or_odor(cmd, tok[1])) { Serial.println("ERR: Bad AIR/ODOR target or level"); return; }
    do_send();
    Serial.println("ALIAS OK");
    return;
  }

  // ---- Aliases: CLEAN / ODOR for switches ----
  if (cmd == "CLEAN" || cmd == "ODOR") {
    if (ntok < 2) { Serial.println("ERR: Missing switch (SV1 | SV2)"); return; }
    if (!apply_clean_or_odor_switch(cmd, tok[1])) { Serial.println("ERR: Bad switch name"); return; }
    do_send();
    Serial.println("ALIAS OK");
    return;
  }

  // ---- Manual ON/OFF/TOGGLE list ----
  bool isToggle=false, setVal=false;
  if (cmd == "TOGGLE") { isToggle = true; }
  else if (cmd == "ON") { setVal = true; }
  else if (cmd == "OFF") { setVal = false; }
  else {
    Serial.println("ERR: Unknown command. Type HELP.");
    return;
  }

  if (ntok < 2) {
    Serial.println("ERR: No targets. Usage: ON|OFF|TOGGLE <name> [name...]");
    return;
  }

  bool ok = true;
  for (int i = 1; i < ntok; ++i) {
    bool got = set_one(tok[i], setVal, isToggle);
    if (!got) {
      Serial.print("ERR: Bad name '"); Serial.print(tok[i]); Serial.println("'");
      ok = false;
    }
  }
  if (ok) { Serial.print(cmd); Serial.println(" OK"); }
}

// ---------- Setup / Loop ----------
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("=== TPIC6B595 serial-staged controller + aliases ===");
  Serial.println("Type HELP for commands.");

  pinMode(PIN_CS, OUTPUT);   // required for SPI on Teensy
  SPI.begin();

  statusTimer = 0;
}

void loop() {
  // Handle serial commands (line-based)
  static String line;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      handle_command_line(line);
      line = "";
    } else {
      line += c;
      if (line.length() > 120) line.remove(0, line.length()-120); // prevent runaway
    }
  }

  // Periodic state print
  if (statusTimer >= STATUS_MS) {
    statusTimer = 0;
    print_states();
  }
}
