/*
  ============================================================================
  TPIC6B595 Serial-Staged Controller + Clean Macro Layer (Teensy 4.1)
  ----------------------------------------------------------------------------
  Purpose:
    - Maintain a staged bitfield for 6 x TPIC6B595 (48 bits total).
    - Provide line-oriented Serial control for:
        * Base ops: ON/OFF/TOGGLE (staging only), SEND, PRINT, CLEAR, RESET.
        * Macros:
            - Olfactometers: CTRL, OD1..OD5 on {OLF1|OLF2|ALL} (auto-SEND).
            - Switches: CLN/ODR on {SV1|SV2|ALL} with optional <ms> timed flip
              to the alternate state (non-blocking; auto-SEND now and on expiry).
        * TEST macro: walking “one-hot” across valid outputs only, with configurable
          ON/OFF durations (non-blocking).
    - NEW: While TEST is active, it **suppresses ALL other transmissions**. Only
      the TEST walker may send frames. Staged state edits continue but won’t transmit
      until TEST STOP.

  Hardware assumptions:
    - SPI MOSI=11, SCK=13. Teensy requires pin 10 as OUTPUT for SPI enable.
    - Daisy-chain order (nearest → farthest):
        OLF1_LO, OLF1_HI, OLF2_LO, OLF2_HI, SV2, SV1
      We build 6 logical bytes and send farthest-first per 595 chains.
    - RCK (latch) is provided by your DAQ (not by this sketch).

  Command Summary
  ---------------
  Base (staging only unless noted):
    ON <name> [name...]       e.g. ON OLF1_0 OLF2_6 SV1_1
    OFF <name> [name...]      e.g. OFF OLF1_3 SV2_0
    TOGGLE <name> [name...]
    SEND                      (transmit current staged state via SPI; suppressed during TEST)
    PRINT                     (print staged bitstring once)
    CLEAR                     (set all staged bits to 0; no send)
    RESET                     (CLEAR, then SEND; SEND suppressed during TEST)
    HELP

    Names:
      OLF1_0..OLF1_11  (12 bits)
      OLF2_0..OLF2_11  (12 bits)
      SV1_0..SV1_1     (2 bits)
      SV2_0..SV2_1     (2 bits)

  Macros (auto-SEND; transmissions suppressed during TEST):
    CTRL <OLF1|OLF2|ALL>      -> AIR: valves 0 & 1 ON (others OFF)
    OD1..OD5 <OLF1|OLF2|ALL>  -> paired valves:
                                 OD1={2,3}, OD2={4,5}, OD3={6,7}, OD4={8,9}, OD5={10,11}
    CLN <SV1|SV2|ALL> [ms]    -> set switch(es) CLEAN (00) now; if ms given, flip to ODR (11)
                                 after <ms> (non-blocking) and SEND again (suppressed during TEST).
    ODR <SV1|SV2|ALL> [ms]    -> set switch(es) ODOR  (11) now; if ms given, flip to CLN (00)
                                 after <ms> and SEND again (suppressed during TEST).

  TEST macro (non-blocking, direct frames; staged state is not modified):
    TEST START                 -> begin walking valid bits with default 500 ms ON / 500 ms OFF
    TEST START <on_ms> <off_ms>-> begin walking with custom durations
    TEST STOP                  -> stop walking immediately
    Valid bits walked: 0–11 (OLF1), 16–27 (OLF2), 32,33 (SV2_0..1), 40,41 (SV1_0..1)

  Heartbeat (periodic state print, default OFF):
    HB ON                    -> enable periodic printing
    HB OFF                   -> disable periodic printing
    HB <ms>                  -> set interval (e.g. HB 2000 for 2 s); also enables
    Format: "000000000000 000000000000 00 00"
              ^ OLF1(12)    ^ OLF2(12)   ^SV1 ^SV2   (MSB..LSB per group)

  Notes:
    - SPI is 1 MHz, MODE0, MSB first. No SD logging here (serial-centric tool).
    - While TEST is active, **all non-TEST sends are suppressed**. Manual SEND and
      auto-SENDs from macros/timers will print a suppression message and do nothing.
  ============================================================================
*/

#include <Arduino.h>
#include <SPI.h>

// ---------------- SPI configuration ----------------
constexpr uint32_t SPI_HZ = 1'000'000;
constexpr int PIN_CS = 10;  // Teensy SPI requirement

// ---------------- Chain layout ---------------------
// Physical order (nearest → farthest): OLF1_LO, OLF1_HI, OLF2_LO, OLF2_HI, SV2, SV1
enum FrameSlot : uint8_t { SLOT_OLF1_HI=0, SLOT_OLF1_LO, SLOT_OLF2_HI, SLOT_OLF2_LO, SLOT_SV1, SLOT_SV2 };
uint8_t FRAME_SEND_ORDER[6] = {
  // out[0] → nearest, out[5] → farthest (sent first)
  SLOT_OLF1_LO,  // out[0]
  SLOT_OLF1_HI,  // out[1]
  SLOT_OLF2_LO,  // out[2]
  SLOT_OLF2_HI,  // out[3]
  SLOT_SV2,      // out[4]
  SLOT_SV1       // out[5]
};

// ---------------- Base staged state ----------------
uint16_t OLF1_bits = 0;  // 12 LSBs: OLF1_0..11
uint16_t OLF2_bits = 0;  // 12 LSBs: OLF2_0..11
uint8_t  SV1_bits  = 0;  // 2 LSBs : SV1_0..1
uint8_t  SV2_bits  = 0;  // 2 LSBs : SV2_0..1

// ---------------- Timed CLN/ODR flips --------------
// Independent non-blocking timers for SV1 and SV2.
struct SwitchTimer {
  bool     active     = false;
  bool     flipToODR  = false;   // true->set to ODR(11) at due; false->set to CLN(00)
  uint32_t due_ms     = 0;       // millis() absolute deadline
};
SwitchTimer svTimer[2]; // [0]=SV1, [1]=SV2

// ---------------- TEST walker (non-blocking) -------
// Valid bit list: 0..11, 16..27, 32,33, 40,41
constexpr uint8_t kAllowedBits[] = {
  0,1,2,3,4,5,6,7,8,9,10,11,
  16,17,18,19,20,21,22,23,24,25,26,27,
  32,33,
  40,41
};
constexpr size_t kAllowedCount = sizeof(kAllowedBits) / sizeof(kAllowedBits[0]);

struct TestRunner {
  bool         active     = false;    // if true, walker is running
  bool         onPhase    = true;     // ON then OFF
  size_t       idx        = 0;        // index into kAllowedBits[]
  uint16_t     bit        = kAllowedBits[0];
  uint32_t     on_ms      = 500;      // default ON duration
  uint32_t     off_ms     = 500;      // default OFF duration
  elapsedMillis phaseTimer;
} testRun;

// ---------------- Periodic heartbeat (default OFF) -
bool          heartbeatOn = false;
uint32_t      heartbeatInterval = 1000;   // ms, configurable
elapsedMillis statusTimer;

// ===================================================
// ================ BASE LAYER =======================
// ===================================================

static inline void build_send_buffer(uint8_t out[6]) {
  uint8_t slots[6] = {0,0,0,0,0,0};

  // 12-bit -> split into LO (bits 0..7) and HI nibble (bits 8..11)
  uint8_t OLF1_LO = (uint8_t)(OLF1_bits & 0xFF);
  uint8_t OLF1_HI = (uint8_t)((OLF1_bits >> 8) & 0x0F);
  uint8_t OLF2_LO = (uint8_t)(OLF2_bits & 0xFF);
  uint8_t OLF2_HI = (uint8_t)((OLF2_bits >> 8) & 0x0F);

  slots[SLOT_OLF1_LO] = OLF1_LO;
  slots[SLOT_OLF1_HI] = OLF1_HI;
  slots[SLOT_OLF2_LO] = OLF2_LO;
  slots[SLOT_OLF2_HI] = OLF2_HI;
  slots[SLOT_SV1]     = (uint8_t)(SV1_bits & 0x03);
  slots[SLOT_SV2]     = (uint8_t)(SV2_bits & 0x03);

  for (int i = 0; i < 6; ++i) out[i] = slots[ FRAME_SEND_ORDER[i] ];
}

static inline void spi_send_48(const uint8_t out[6]) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  for (int i = 5; i >= 0; --i) SPI.transfer(out[i]); // farthest first
  SPI.endTransaction();
}

// Gate to check if non-TEST transmissions are allowed
static inline bool tx_allowed() { return !testRun.active; }

static inline void do_send() {
  if (!tx_allowed()) {
    Serial.println("SEND suppressed (TEST active)");
    return;
  }

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
  char line[64]; int p = 0;

  for (int b = 11; b >= 0; --b) line[p++] = ((OLF1_bits >> b) & 1) ? '1':'0';
  line[p++] = ' ';
  for (int b = 11; b >= 0; --b) line[p++] = ((OLF2_bits >> b) & 1) ? '1':'0';
  line[p++] = ' ';
  for (int b = 1; b >= 0; --b)  line[p++] = ((SV1_bits  >> b) & 1) ? '1':'0';
  line[p++] = ' ';
  for (int b = 1; b >= 0; --b)  line[p++] = ((SV2_bits  >> b) & 1) ? '1':'0';

  line[p] = 0;
  Serial.println(line);
}

static inline void clear_states() { OLF1_bits = OLF2_bits = 0; SV1_bits = SV2_bits = 0; }

// Base bit mutators (staging only)
static inline bool apply_bit(uint16_t &field, int idx, int maxBits, int op) {
  if (idx < 0 || idx >= maxBits) return false;
  uint16_t m = (1u << idx);
  if (op > 0) field |= m;
  else if (op < 0) field &= ~m;
  else field ^= m;
  return true;
}
static inline bool apply_bit2(uint8_t &field, int idx, int maxBits, int op) {
  if (idx < 0 || idx >= maxBits) return false;
  uint8_t m = (1u << idx);
  if (op > 0) field |= m;
  else if (op < 0) field &= ~m;
  else field ^= m;
  return true;
}

static inline bool set_one(const String& key, int op /*+1 on, -1 off, 0 toggle*/) {
  String s = key; s.toUpperCase();
  int us = s.indexOf('_');
  if (us < 0) return false;
  String name = s.substring(0, us);
  int idx     = s.substring(us+1).toInt();

  if (name == "OLF1") return apply_bit (OLF1_bits, idx, 12, op);
  if (name == "OLF2") return apply_bit (OLF2_bits, idx, 12, op);
  if (name == "SV1")  return apply_bit2(SV1_bits,  idx,  2, op);
  if (name == "SV2")  return apply_bit2(SV2_bits,  idx,  2, op);
  return false;
}

// Olfactometer helpers
static inline void olf_set_pair(uint16_t &field, uint8_t firstBit /*0,2,4,6,8,10*/) {
  field = 0;
  if (firstBit < 12) field |= (1u << firstBit);
  if (firstBit + 1 < 12) field |= (1u << (firstBit + 1));
}

// ===================================================
// ================ MACRO LAYER ======================
// ===================================================

// CTRL -> firstBit=0; OD1..OD5 -> firstBit=2,4,6,8,10
static inline bool parse_olf_macro(const String& ucmd, uint8_t &firstBitOut) {
  if (ucmd == "CTRL") { firstBitOut = 0; return true; }
  // Strictly accept OD + single digit 1..5
  if (ucmd.length() == 3 && ucmd[0]=='O' && ucmd[1]=='D' && ucmd[2]>='1' && ucmd[2]<='5') {
    firstBitOut = (uint8_t)(2 * (ucmd[2] - '0'));  // '1'->2, '5'->10
    return true;
  }
  return false; // Reject ODR here explicitly
}

static inline bool apply_olf_macro_and_send(const String& ucmd, const String& target) {
  uint8_t first = 0;
  if (!parse_olf_macro(ucmd, first)) return false;

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
  do_send();   // may be suppressed during TEST
  return true;
}

// UPDATED: CLN/ODR now accept SV1 | SV2 | ALL
static inline bool apply_switch_macro(const String& ucmd, const String& which, long ms /*-1 if none*/) {
  bool toClean = (ucmd == "CLN");
  bool toOdor  = (ucmd == "ODR");
  if (!toClean && !toOdor) return false;

  String w = which; w.toUpperCase();
  bool doSV1 = false, doSV2 = false;

  if (w == "SV1") { doSV1 = true; }
  else if (w == "SV2") { doSV2 = true; }
  else if (w == "ALL") { doSV1 = doSV2 = true; }
  else return false;

  if (doSV1) SV1_bits = toClean ? 0b00 : 0b11;
  if (doSV2) SV2_bits = toClean ? 0b00 : 0b11;

  do_send();   // may be suppressed during TEST

  if (ms > 0) {
    if (doSV1) {
      svTimer[0].active    = true;
      svTimer[0].flipToODR = toClean;  // If we set CLN now, flip to ODR later; if ODR now, flip to CLN later.
      svTimer[0].due_ms    = millis() + (uint32_t)ms;
    }
    if (doSV2) {
      svTimer[1].active    = true;
      svTimer[1].flipToODR = toClean;
      svTimer[1].due_ms    = millis() + (uint32_t)ms;
    }
  }
  return true;
}

// ---------------- TEST helpers ---------------------
// Send a direct “all zeros” frame (does not alter staged state).
static inline void test_send_all_zero() {
  uint8_t out[6] = {0,0,0,0,0,0};
  spi_send_48(out);
  Serial.println("TEST: OFF gap (all zeros)");
}

// Send a direct one-hot frame for absolute bit index (0..47); does not alter staged state.
static inline void test_send_one_hot(uint16_t absBit) {
  // Build logical slots directly (without touching staged bits)
  uint8_t slots[6] = {0,0,0,0,0,0};
  uint8_t byteIdx  = absBit / 8;       // 0..5
  uint8_t bitInByte= absBit % 8;       // 0..7

  // byteIdx mapping: 0=OLF1_LO, 1=OLF1_HI, 2=OLF2_LO, 3=OLF2_HI, 4=SV2, 5=SV1
  const uint8_t byteIdx_to_slot[6] = {
    SLOT_OLF1_LO, SLOT_OLF1_HI, SLOT_OLF2_LO, SLOT_OLF2_HI, SLOT_SV2, SLOT_SV1
  };
  slots[ byteIdx_to_slot[byteIdx] ] = (uint8_t)(1u << bitInByte);

  // Map to out buffer by daisy-chain order
  uint8_t out[6];
  for (int i = 0; i < 6; ++i) out[i] = slots[ FRAME_SEND_ORDER[i] ];

  // Transmit
  spi_send_48(out);

  // Print for visibility
  Serial.print("TEST: ON bit ");
  Serial.print(absBit);
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

// ===================================================
// ================ COMMAND PARSER ===================
// ===================================================

static inline bool parse_int_if_numeric(const String& s, long &out) {
  if (s.length() == 0) return false;
  for (size_t i = 0; i < (size_t)s.length(); ++i) {
    char c = s.charAt(i);
    if (c < '0' || c > '9') return false;
  }
  out = s.toInt();
  return true;
}

// Signed/unsigned warning fix: cache line.length() as int.
static inline void handle_command_line(String line) {
  line.trim();
  if (line.length() == 0) return;

  const int MAX_TOK = 32;
  String tok[MAX_TOK];
  int ntok = 0;

  int start = 0;
  const int lineLen = (int)line.length();
  while (start < lineLen && ntok < MAX_TOK) {
    int sp = line.indexOf(' ', start);
    if (sp < 0) sp = lineLen;
    if (sp > start) tok[ntok++] = line.substring(start, sp);
    start = sp + 1;
  }
  if (ntok == 0) return;

  // Upper-cased command for comparisons
  String cmd = tok[0]; cmd.toUpperCase();

  // ---- HELP ----
  if (cmd == "HELP") {
    Serial.println(F(
      "Base:\n"
      "  ON <name> [name...]      (staged)\n"
      "  OFF <name> [name...]     (staged)\n"
      "  TOGGLE <name> [name...]  (staged)\n"
      "  CLEAR                    (no send)\n"
      "  RESET                    (clear then send; SEND suppressed during TEST)\n"
      "  SEND                     (suppressed during TEST)\n"
      "  PRINT\n"
      "Names: OLF1_0..OLF1_11, OLF2_0..OLF2_11, SV1_0..SV1_1, SV2_0..SV2_1\n"
      "\n"
      "Macros (auto-SEND; transmissions suppressed during TEST):\n"
      "  CTRL <OLF1|OLF2|ALL>\n"
      "  OD1..OD5 <OLF1|OLF2|ALL>\n"
      "  CLN <SV1|SV2|ALL> [ms]\n"
      "  ODR <SV1|SV2|ALL> [ms]\n"
      "\n"
      "TEST (non-blocking; suppresses all other sends):\n"
      "  TEST START               (500 ms ON / 500 ms OFF)\n"
      "  TEST START <on_ms> <off_ms>\n"
      "  TEST STOP\n"
      "\n"
      "Heartbeat (periodic state print, default OFF):\n"
      "  HB ON                    enable\n"
      "  HB OFF                   disable\n"
      "  HB <ms>                  set interval & enable\n"
      "Format: 000000000000 000000000000 00 00\n"
    ));
    return;
  }

  // ---- PRINT / CLEAR / RESET / SEND ----
  if (cmd == "PRINT") { Serial.print("STATE: "); print_states(); return; }
  if (cmd == "CLEAR") { clear_states(); Serial.println("CLEARED"); return; }
  if (cmd == "RESET") { clear_states(); do_send(); Serial.println("RESET OK"); return; }
  if (cmd == "SEND")  { do_send(); return; }

  // ---- Switch macros FIRST: disambiguate 'ODR' from 'ODx' ----
  if (cmd == "CLN" || cmd == "ODR") {
    if (ntok < 2) { Serial.println("ERR: Missing switch (SV1 | SV2 | ALL)"); return; }
    long ms = -1;
    if (ntok >= 3) (void)parse_int_if_numeric(tok[2], ms);
    if (!apply_switch_macro(cmd, tok[1], ms)) {
      Serial.println("ERR: Bad CLN/ODR args");
    } else {
      Serial.print(cmd); Serial.print(" "); Serial.print(tok[1]);
      if (ms > 0) { Serial.print(" OK (timed "); Serial.print(ms); Serial.println(" ms)"); }
      else        { Serial.println(" OK"); }
    }
    return;
  }

  // ---- Olfactometer macros: strictly CTRL or OD[1-5] ----
  if (cmd == "CTRL" || (cmd.length()==3 && cmd[0]=='O' && cmd[1]=='D' && cmd[2]>='1' && cmd[2]<='5')) {
    if (ntok < 2) { Serial.println("ERR: Missing target (use OLF1 | OLF2 | ALL)"); return; }
    if (!apply_olf_macro_and_send(cmd, tok[1])) {
      Serial.println("ERR: Bad CTRL/ODx target or level");
    } else {
      Serial.print(cmd); Serial.print(" "); Serial.print(tok[1]); Serial.println(" OK");
    }
    return;
  }

  // ---- Heartbeat control ----
  if (cmd == "HB") {
    if (ntok < 2) { Serial.println("ERR: HB requires ON, OFF, or <ms>"); return; }
    String arg = tok[1]; arg.toUpperCase();
    if (arg == "ON")       { heartbeatOn = true;  Serial.println("HB ON"); }
    else if (arg == "OFF") { heartbeatOn = false; Serial.println("HB OFF"); }
    else {
      long v;
      if (parse_int_if_numeric(tok[1], v) && v > 0) {
        heartbeatInterval = (uint32_t)v;
        heartbeatOn = true;
        Serial.print("HB interval="); Serial.print(v); Serial.println(" ms (enabled)");
      } else {
        Serial.println("ERR: HB expects ON, OFF, or positive <ms>");
      }
    }
    return;
  }

  // ---- TEST macro ----
  if (cmd == "TEST") {
    if (ntok < 2) { Serial.println("ERR: TEST requires START or STOP"); return; }
    String sub = tok[1]; sub.toUpperCase();

    if (sub == "START") {
      // Parse optional on/off ms
      uint32_t onms = 500, offms = 500;
      if (ntok >= 3) {
        long v; if (parse_int_if_numeric(tok[2], v) && v >= 0) onms = (uint32_t)v;
      }
      if (ntok >= 4) {
        long v; if (parse_int_if_numeric(tok[3], v) && v >= 0) offms = (uint32_t)v;
      }

      testRun.on_ms   = onms;
      testRun.off_ms  = offms;
      testRun.idx     = 0;
      testRun.bit     = kAllowedBits[0];
      testRun.onPhase = true;
      testRun.phaseTimer = 0;
      testRun.active  = true;

      // Kick off first ON frame immediately
      test_send_one_hot(testRun.bit);

      Serial.print("TEST START (on=");
      Serial.print(onms);
      Serial.print(" ms, off=");
      Serial.print(offms);
      Serial.println(" ms) — transmissions from other commands are suppressed.");
      return;
    }
    else if (sub == "STOP") {
      testRun.active = false;
      // Restore staged state to hardware (TEST doesn't modify staged state,
      // so this clears any residual one-hot frame left by the walker).
      do_send();
      Serial.println("TEST STOP — staged state restored; transmissions re-enabled.");
      return;
    }
    else {
      Serial.println("ERR: TEST expects START or STOP");
      return;
    }
  }

  // ---- Manual staging: ON/OFF/TOGGLE ----
  int op;
  if      (cmd == "ON")     op = +1;
  else if (cmd == "OFF")    op = -1;
  else if (cmd == "TOGGLE") op =  0;
  else { Serial.println("ERR: Unknown command. Type HELP."); return; }

  if (ntok < 2) { Serial.println("ERR: No targets. Usage: ON|OFF|TOGGLE <name> [name...]"); return; }

  bool ok = true;
  for (int i = 1; i < ntok; ++i) {
    if (!set_one(tok[i], op)) {
      Serial.print("ERR: Bad name '"); Serial.print(tok[i]); Serial.println("'");
      ok = false;
    }
  }
  if (ok) { Serial.print(cmd); Serial.println(" OK (staged)"); }
}

// ===================================================
// ================ SETUP / LOOP =====================
// ===================================================

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("=== TPIC6B595 serial-staged controller + macros (CTRL/OD1..OD5, CLN/ODR timed, TEST walker) ===");
  Serial.println("Type HELP for commands.");

  pinMode(PIN_CS, OUTPUT);   // required for SPI on Teensy
  SPI.begin();

  statusTimer = 0;
}

void loop() {
  // --------- serial line reader ---------
  static String line;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      handle_command_line(line);
      line = "";
    } else {
      line += c;
      if (line.length() > 200) line.remove(0, line.length()-200);
    }
  }

  // --------- timed CLN/ODR flips (non-blocking) ---------
  uint32_t now = millis();
  for (int i = 0; i < 2; ++i) {
    if (svTimer[i].active && (int32_t)(now - svTimer[i].due_ms) >= 0) {
      if (i == 0) SV1_bits = svTimer[i].flipToODR ? 0b11 : 0b00;
      else        SV2_bits = svTimer[i].flipToODR ? 0b11 : 0b00;
      do_send();                  // may be suppressed during TEST
      svTimer[i].active = false;
    }
  }

  // --------- TEST walker (non-blocking) ---------------
  if (testRun.active) {
    if (testRun.onPhase) {
      if (testRun.phaseTimer >= testRun.on_ms) {
        // Move to OFF gap
        testRun.onPhase = false;
        testRun.phaseTimer = 0;
        test_send_all_zero();
      }
    } else { // OFF phase
      if (testRun.phaseTimer >= testRun.off_ms) {
        // Advance to next allowed bit and send ON
        testRun.onPhase = true;
        testRun.phaseTimer = 0;
        testRun.idx = (testRun.idx + 1) % kAllowedCount;
        testRun.bit = kAllowedBits[testRun.idx];
        test_send_one_hot(testRun.bit);
      }
    }
  }

  // --------- periodic heartbeat (if enabled) ---------
  if (heartbeatOn && statusTimer >= heartbeatInterval) {
    statusTimer = 0;
    print_states();
  }
}
