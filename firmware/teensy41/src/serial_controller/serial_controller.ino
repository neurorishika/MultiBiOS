/*
  ============================================================================
  TPIC6B595 Serial-Staged Controller  v2  (Teensy 4.1)
  ----------------------------------------------------------------------------
  What changed from v1
  --------------------
  ARCHITECTURE
    - Zero heap: no String class anywhere. All parsing uses strtok on a static
      char buffer. No malloc, no fragmentation, no latency spikes.
    - Precomputed frame cache: staged bits → 6-byte frame is rebuilt only when
      the bits actually change (dirty flag). SPI path is always a memcpy + raw
      transfer with no arithmetic.
    - Non-suppressive concurrency: TEST and SEQ no longer block other commands.
      Any new command immediately cancels the running walker/sequence, applies,
      and transmits. Fast overwrite is the default.

  NEW FEATURES
    PULSE <names...> <ms>
        Turn named bits ON for <ms>, then auto-restore the pre-pulse frame.
        Up to MAX_PULSE_SLOTS independent overlapping pulses. New pulses on
        the same bits cancel the older slot and restart cleanly.

    SEQ (sequence engine)
        Pre-bake named frame sequences; trigger by name; loop or one-shot.
        Steps accumulate state (CLR resets cursor). Pre-baked at definition
        time so runtime is just a frame copy + SPI transfer.
          SEQ NEW  <name>
          SEQ ADD  <cmd> [args...] <dur_ms>   (sub-commands: CLR CTRL OD1-5
                                               CLN ODR ON OFF FRAME)
          SEQ END
          SEQ RUN  <name> [count]             (count -1 = loop forever)
          SEQ STOP
          SEQ LIST
          SEQ DEL  <name>

    PRESET (named frame snapshots)
          PRESET SAVE <name>
          PRESET LOAD <name>
          PRESET LIST
          PRESET DEL  <name>

    FRAME <h0> <h1> <h2> <h3> <h4> <h5>
        Set all six bytes directly as hex and SEND immediately.

    STATUS
        Full verbose dump: staged bits, active pulse slots, running
        sequence, test walker state, preset list, SPI speed.

    WATCH ON|OFF
        Print a "WATCH: <bitstring>" line whenever the hardware frame
        actually changes (works for staged sends, pulses, seqs, TEST).

    TEST improvements
        TEST ONCE           — single pass then stop
        TEST RANGE <a> <b> [on_ms [off_ms]]
                            — walk only valid-bit indices a..b
        TEST STEP           — manual advance by one bit (while paused)

    SPD <hz>
        Change SPI clock at runtime (e.g. SPD 4000000).

    CMD tagging
        Prefix any command with @<N> (a positive integer tag):
          @42 OD1 OLF1
        All reply lines for that command are prefixed "@42 ".
        Host software can pipeline commands and correlate replies.

    BINARY PROTOCOL (compact machine interface)
        If the first byte of a line is 0xAA the rest is a binary packet:
          [0xAA][cmd:1][len:1][payload:len][crc8:1]
        CRC-8 covers cmd+payload (NOT the len byte).
        Commands:
          0x01  SEND_FRAME   payload=6 bytes raw; sends immediately
          0x02  PULSE_FRAME  payload=6 bytes + uint32_t ms (LE)
          0x03  QUERY        payload=none; replies [0xAB][0x03][6 bytes frame]
          0x04  SEQ_RUN      payload=name (null-term or len-terminated)
          0x05  SEQ_STOP     payload=none
          0x06  PRESET_LOAD  payload=name
          0x07  PRESET_SAVE  payload=name
        Reply: [0xAB][status:1] where 0=OK, non-zero=error, except QUERY.

  COMMAND SUMMARY  (all ASCII, case-insensitive)
  -----------------------------------------------
  Base (staging):
    ON   <name> [name...]      Stage bits high
    OFF  <name> [name...]      Stage bits low
    TOGGLE <name> [name...]    Stage bits flipped
    SEND                       Push staged frame to SPI
    PRINT / STATE              Print staged bitstring
    CLEAR                      Zero staging (no send)
    RESET                      CLEAR + SEND
    FRAME <h0>..<h5>           Set raw hex frame + SEND

  Macros (auto-SEND; cancel SEQ/TEST):
    CTRL  <OLF1|OLF2|ALL>
    OD1..OD5 <OLF1|OLF2|ALL>
    CLN   <SV1|SV2|ALL> [ms]
    ODR   <SV1|SV2|ALL> [ms]

  Pulse (auto-restore, non-blocking):
    PULSE <name> [name...] <ms>
    PULSE?                     List active pulse slots

  Sequences (pre-baked, non-blocking):
    SEQ NEW  <name>
    SEQ ADD  CLR <ms>
    SEQ ADD  CTRL|OD1-5 <target> <ms>
    SEQ ADD  CLN|ODR <target> <ms>
    SEQ ADD  ON|OFF <names...> <ms>
    SEQ ADD  FRAME <h0>..<h5> <ms>
    SEQ END
    SEQ RUN  <name> [count]     count: n (n passes), -1 (loop)
    SEQ STOP
    SEQ LIST
    SEQ DEL  <name>

  Presets:
    PRESET SAVE <name>
    PRESET LOAD <name>
    PRESET LIST
    PRESET DEL  <name>

  Test walker:
    TEST START [on_ms [off_ms]]
    TEST ONCE  [on_ms [off_ms]]
    TEST RANGE <a> <b> [on_ms [off_ms]]
    TEST STEP
    TEST STOP

  Config:
    HB ON|OFF|<ms>             Heartbeat (default OFF)
    WATCH ON|OFF               Print on frame change
    SPD <hz>                   SPI clock speed
    STATUS                     Full verbose dump
    HELP

  Bit names:
    OLF1_0..OLF1_11   OLF2_0..OLF2_11   SV1_0..SV1_1   SV2_0..SV2_1

  Hardware:
    SPI MOSI=11, SCK=13, CS=10.
    Chain (nearest→farthest): OLF1_LO OLF1_HI OLF2_LO OLF2_HI SV2 SV1
    RCK latch driven externally by DAQ.
  ============================================================================
*/

#include <Arduino.h>
#include <SPI.h>
#include <string.h>
#include <stdlib.h>

// ============================================================
// ========================= CONFIG ===========================
// ============================================================
#define MAX_SEQ          8
#define MAX_SEQ_STEPS   32
#define MAX_SEQ_NAME    16
#define MAX_PRESETS      8
#define MAX_PRESET_NAME 16
#define MAX_PULSE_SLOTS 16
#define MAX_CMD_BUF    256
#define MAX_TOKENS      28
#define MAX_BIN_PAYLOAD 64
#define SPI_HZ_DEFAULT  1000000UL

// ============================================================
// ==================== FRAME ENCODING ========================
// ============================================================
// stagedFrame[] index:
//  0 = OLF1_LO  (OLF1 bits 0-7)
//  1 = OLF1_HI  (OLF1 bits 8-11, lower nibble)
//  2 = OLF2_LO  (OLF2 bits 0-7)
//  3 = OLF2_HI  (OLF2 bits 8-11, lower nibble)
//  4 = SV2      (bits 0-1)
//  5 = SV1      (bits 0-1)
// SPI wire order: index 5 first (farthest IC), index 0 last.
//
// Absolute bit space used by TEST/PULSE name→bit helpers:
//  0-11  = OLF1_0..11  (maps to f[0] bits0-7 and f[1] bits0-3)
//  16-27 = OLF2_0..11  (maps to f[2] bits0-7 and f[3] bits0-3)
//  32-33 = SV2_0..1    (f[4] bits 0-1)
//  40-41 = SV1_0..1    (f[5] bits 0-1)
constexpr int PIN_CS = 10;

// ============================================================
// ===================== STAGED STATE =========================
// ============================================================
struct BitState {
  uint16_t olf1 = 0;
  uint16_t olf2 = 0;
  uint8_t  sv1  = 0;
  uint8_t  sv2  = 0;
};

BitState staged;
uint8_t  stagedFrame[6];
bool     frameDirty   = true;
uint8_t  lastSentFrame[6];
uint32_t g_spi_hz     = SPI_HZ_DEFAULT;

static inline void bits_to_frame(const BitState& b, uint8_t f[6]) {
  f[0] = (uint8_t)(b.olf1 & 0xFF);
  f[1] = (uint8_t)((b.olf1 >> 8) & 0x0F);
  f[2] = (uint8_t)(b.olf2 & 0xFF);
  f[3] = (uint8_t)((b.olf2 >> 8) & 0x0F);
  f[4] = (uint8_t)(b.sv2  & 0x03);
  f[5] = (uint8_t)(b.sv1  & 0x03);
}
static inline void frame_to_bits(const uint8_t f[6], BitState& b) {
  b.olf1 = f[0] | ((uint16_t)(f[1] & 0x0F) << 8);
  b.olf2 = f[2] | ((uint16_t)(f[3] & 0x0F) << 8);
  b.sv2  = f[4] & 0x03;
  b.sv1  = f[5] & 0x03;
}
static inline void rebuild_staged_frame() {
  bits_to_frame(staged, stagedFrame);
  frameDirty = false;
}
static inline void mark_dirty() { frameDirty = true; }

// ============================================================
// ======================== SPI LAYER =========================
// ============================================================
static inline void spi_send_raw(const uint8_t f[6]) {
  SPI.beginTransaction(SPISettings(g_spi_hz, MSBFIRST, SPI_MODE0));
  for (int i = 5; i >= 0; --i) SPI.transfer(f[i]);
  SPI.endTransaction();
}

// ============================================================
// =================== PULSE TIMER POOL =======================
// ============================================================
struct PulseSlot {
  bool    active        = false;
  uint32_t due_ms       = 0;
  uint8_t restoreFrame[6];
};
PulseSlot pulseSlots[MAX_PULSE_SLOTS];

static int find_free_pulse() {
  for (int i = 0; i < MAX_PULSE_SLOTS; ++i)
    if (!pulseSlots[i].active) return i;
  return -1;
}
static void cancel_all_pulses() {
  for (int i = 0; i < MAX_PULSE_SLOTS; ++i) pulseSlots[i].active = false;
}

// ============================================================
// ===================== PRESET STORAGE =======================
// ============================================================
struct Preset {
  bool   used = false;
  char   name[MAX_PRESET_NAME];
  uint8_t frame[6];
};
Preset presets[MAX_PRESETS];

static int preset_find(const char* name) {
  for (int i = 0; i < MAX_PRESETS; ++i)
    if (presets[i].used && strcmp(presets[i].name, name) == 0) return i;
  return -1;
}
static int preset_free_slot() {
  for (int i = 0; i < MAX_PRESETS; ++i)
    if (!presets[i].used) return i;
  return -1;
}

// ============================================================
// ==================== SEQUENCE ENGINE =======================
// ============================================================
struct SeqStep {
  uint8_t  frame[6];
  uint32_t dur_ms;
};
struct Sequence {
  bool    defined = false;
  char    name[MAX_SEQ_NAME];
  uint8_t nsteps  = 0;
  SeqStep steps[MAX_SEQ_STEPS];
};
Sequence sequences[MAX_SEQ];

struct SeqRunner {
  bool          active    = false;
  uint8_t       seqIdx    = 0;
  uint8_t       stepIdx   = 0;
  int32_t       remaining = 1;   // passes left; -1 = loop
  elapsedMillis stepTimer;
  bool          stepSent  = false;
  uint8_t       restoreFrame[6]; // staged frame at SEQ RUN time
};
SeqRunner seqRunner;

// Definition cursor (used during SEQ NEW .. SEQ END)
struct DefCursor {
  bool    active  = false;
  uint8_t seqIdx  = 0xFF;
  BitState bits;
};
DefCursor defCursor;

static int seq_find(const char* name) {
  for (int i = 0; i < MAX_SEQ; ++i)
    if (sequences[i].defined && strcmp(sequences[i].name, name) == 0) return i;
  return -1;
}
static int seq_free_slot() {
  for (int i = 0; i < MAX_SEQ; ++i)
    if (!sequences[i].defined) return i;
  return -1;
}

// ============================================================
// ====================== TEST WALKER =========================
// ============================================================
constexpr uint8_t kTestBits[] = {
  0,1,2,3,4,5,6,7,8,9,10,11,
  16,17,18,19,20,21,22,23,24,25,26,27,
  32,33,
  40,41
};
constexpr uint8_t kTestBitCount = sizeof(kTestBits) / sizeof(kTestBits[0]);

struct TestWalker {
  bool          active   = false;
  bool          oneShot  = false;
  bool          onPhase  = true;
  uint8_t       idxStart = 0;
  uint8_t       idxEnd   = kTestBitCount - 1;
  uint8_t       idx      = 0;
  uint32_t      on_ms    = 500;
  uint32_t      off_ms   = 500;
  elapsedMillis phaseTimer;
  uint8_t       restoreFrame[6];
};
TestWalker testWalker;

// ============================================================
// ==================== HB / WATCH / TAG ======================
// ============================================================
bool          heartbeatOn       = false;
uint32_t      heartbeatInterval = 1000;
elapsedMillis heartbeatTimer;
bool          watchOn           = false;

static char g_tag[12] = "";  // command reply tag, empty=none

static inline void tag_print() {
  if (g_tag[0]) { Serial.print(g_tag); Serial.print(' '); }
}

// ============================================================
// ==================== OUTPUT HELPERS ========================
// ============================================================
static void print_state_line(const uint8_t f[6]) {
  BitState b; frame_to_bits(f, b);
  char buf[34];  int p = 0;
  for (int i = 11; i >= 0; --i) buf[p++] = ((b.olf1 >> i) & 1) ? '1' : '0';
  buf[p++] = ' ';
  for (int i = 11; i >= 0; --i) buf[p++] = ((b.olf2 >> i) & 1) ? '1' : '0';
  buf[p++] = ' ';
  for (int i =  1; i >= 0; --i) buf[p++] = ((b.sv1  >> i) & 1) ? '1' : '0';
  buf[p++] = ' ';
  for (int i =  1; i >= 0; --i) buf[p++] = ((b.sv2  >> i) & 1) ? '1' : '0';
  buf[p] = '\0';
  Serial.println(buf);
}

// Print 6 SPI bytes in wire order (index 5 first) as space-separated hex
static void print_frame_hex(const uint8_t f[6]) {
  for (int i = 5; i >= 0; --i) {
    if (f[i] < 0x10) Serial.print('0');
    Serial.print(f[i], HEX);
    if (i > 0) Serial.print(' ');
  }
  Serial.println();
}

// Central send: SPI + echo + WATCH check
static void do_send_frame(const uint8_t f[6], const char* label) {
  spi_send_raw(f);
  tag_print();
  Serial.print(label);
  Serial.print(" Data: ");
  print_frame_hex(f);
  if (watchOn && memcmp(f, lastSentFrame, 6) != 0) {
    Serial.print("WATCH: ");
    print_state_line(f);
  }
  memcpy(lastSentFrame, f, 6);
}

static void do_send_staged() {
  if (frameDirty) rebuild_staged_frame();
  do_send_frame(stagedFrame, "SEND");
}

// ============================================================
// ===================== BIT MUTATORS =========================
// ============================================================
// name is already uppercased; op: +1=ON, -1=OFF, 0=TOGGLE
static bool set_bit(BitState& b, const char* name, int op) {
  const char* us = strchr(name, '_');
  if (!us) return false;
  // Validate suffix is purely numeric (reject OLF1_abc etc.)
  const char* sfx = us + 1;
  if (!*sfx) return false;  // trailing underscore only
  for (const char* p = sfx; *p; ++p)
    if (*p < '0' || *p > '9') return false;
  int idx = atoi(sfx);
  int nlen = (int)(us - name);
  if      (nlen == 4 && strncmp(name, "OLF1", 4) == 0 && idx >= 0 && idx <= 11) {
    uint16_t m = (1u << idx);
    if (op > 0) b.olf1 |= m; else if (op < 0) b.olf1 &= ~m; else b.olf1 ^= m;
  } else if (nlen == 4 && strncmp(name, "OLF2", 4) == 0 && idx >= 0 && idx <= 11) {
    uint16_t m = (1u << idx);
    if (op > 0) b.olf2 |= m; else if (op < 0) b.olf2 &= ~m; else b.olf2 ^= m;
  } else if (nlen == 3 && strncmp(name, "SV1", 3) == 0 && idx >= 0 && idx <= 1) {
    uint8_t m = (1u << idx);
    if (op > 0) b.sv1 |= m; else if (op < 0) b.sv1 &= ~m; else b.sv1 ^= m;
  } else if (nlen == 3 && strncmp(name, "SV2", 3) == 0 && idx >= 0 && idx <= 1) {
    uint8_t m = (1u << idx);
    if (op > 0) b.sv2 |= m; else if (op < 0) b.sv2 &= ~m; else b.sv2 ^= m;
  } else {
    return false;
  }
  return true;
}

static void olf_set_pair(BitState& b, bool o1, bool o2, uint8_t firstBit) {
  uint16_t v = 0;
  if (firstBit     < 12) v |= (1u << firstBit);
  if (firstBit + 1 < 12) v |= (1u << (firstBit + 1));
  if (o1) b.olf1 = v;
  if (o2) b.olf2 = v;
}

// ============================================================
// ==================== MACRO PARSERS =========================
// ============================================================
static bool parse_olf_cmd(const char* cmd, uint8_t& firstBit) {
  if (strcmp(cmd, "CTRL") == 0) { firstBit = 0; return true; }
  if (strlen(cmd) == 3 && cmd[0]=='O' && cmd[1]=='D'
      && cmd[2] >= '1' && cmd[2] <= '5') {
    firstBit = (uint8_t)(2 * (cmd[2] - '0'));
    return true;
  }
  return false;
}
static bool parse_olf_target(const char* t, bool& o1, bool& o2) {
  if (strcmp(t, "OLF1") == 0) { o1=true;  o2=false; return true; }
  if (strcmp(t, "OLF2") == 0) { o1=false; o2=true;  return true; }
  if (strcmp(t, "ALL")  == 0) { o1=true;  o2=true;  return true; }
  return false;
}
static bool parse_sv_target(const char* t, bool& s1, bool& s2) {
  if (strcmp(t, "SV1") == 0) { s1=true;  s2=false; return true; }
  if (strcmp(t, "SV2") == 0) { s1=false; s2=true;  return true; }
  if (strcmp(t, "ALL") == 0) { s1=true;  s2=true;  return true; }
  return false;
}

// ============================================================
// ================ UTILITIES / CRC ===========================
// ============================================================
static bool is_numeric(const char* s) {
  if (!s || !*s) return false;
  const char* p = s;
  if (*p == '-') ++p;                        // allow leading minus
  if (!*p) return false;
  for (; *p; ++p) if (*p < '0' || *p > '9') return false;
  return true;
}
static void str_upper(char* s) {
  for (; *s; ++s) if (*s >= 'a' && *s <= 'z') *s -= 32;
}
static uint8_t crc8_buf(const uint8_t* d, uint8_t n) {
  uint8_t crc = 0;
  while (n--) { crc ^= *d++; for (uint8_t i=0;i<8;++i) crc=(crc&0x80)?(crc<<1)^0x07:(crc<<1); }
  return crc;
}

// ============================================================
// =========== SEQ STEP COMMAND APPLIER =======================
// ============================================================
// Parses a SEQ ADD sub-command onto a BitState cursor.
// tok[0..ntok-1] are the tokens AFTER "SEQ ADD" and BEFORE the duration.
// Returns false on error.
static bool apply_seq_cmd(BitState& b, char** tok, int ntok) {
  if (ntok < 1) return false;
  if (strcmp(tok[0], "CLR") == 0 || strcmp(tok[0], "CLEAR") == 0) {
    b = {0, 0, 0, 0}; return true;
  }
  uint8_t firstBit;
  if (parse_olf_cmd(tok[0], firstBit)) {
    if (ntok < 2) return false;
    bool o1=false, o2=false;
    if (!parse_olf_target(tok[1], o1, o2)) return false;
    olf_set_pair(b, o1, o2, firstBit); return true;
  }
  if (strcmp(tok[0], "CLN") == 0 || strcmp(tok[0], "ODR") == 0) {
    if (ntok < 2) return false;
    bool toClean = (tok[0][0] == 'C');
    bool s1=false, s2=false;
    if (!parse_sv_target(tok[1], s1, s2)) return false;
    if (s1) b.sv1 = toClean ? 0x00 : 0x03;
    if (s2) b.sv2 = toClean ? 0x00 : 0x03;
    return true;
  }
  if (strcmp(tok[0], "ON") == 0 || strcmp(tok[0], "OFF") == 0) {
    int op = (tok[0][1] == 'N') ? +1 : -1;
    for (int i = 1; i < ntok; ++i)
      if (!set_bit(b, tok[i], op)) return false;
    return true;
  }
  if (strcmp(tok[0], "FRAME") == 0) {
    if (ntok < 7) return false;  // FRAME h0..h5
    uint8_t f[6];
    for (int i = 0; i < 6; ++i) f[i] = (uint8_t)strtoul(tok[i+1], nullptr, 16);
    frame_to_bits(f, b); return true;
  }
  return false;
}

// ============================================================
// ==================== RUNNER STOP HELPERS ===================
// ============================================================
static void stop_test() {
  if (!testWalker.active) return;
  testWalker.active = false;
  // Restore staged frame
  do_send_frame(testWalker.restoreFrame, "TEST STOP restore");
  memcpy(stagedFrame, testWalker.restoreFrame, 6);
  frame_to_bits(testWalker.restoreFrame, staged);
  frameDirty = false;
}
static void stop_seq() {
  if (!seqRunner.active) return;
  seqRunner.active = false;
  do_send_frame(seqRunner.restoreFrame, "SEQ STOP restore");
  memcpy(stagedFrame, seqRunner.restoreFrame, 6);
  frame_to_bits(seqRunner.restoreFrame, staged);
  frameDirty = false;
}
// Stop any autonomous runner before a new command takes effect
static void preempt_runners() {
  if (testWalker.active) { stop_test(); tag_print(); Serial.println("TEST preempted"); }
  if (seqRunner.active)  { stop_seq();  tag_print(); Serial.println("SEQ  preempted"); }
}

// ============================================================
// ==================== TEST HELPERS ==========================
// ============================================================
static void test_send_one_hot(uint8_t absBit) {
  uint8_t f[6] = {0,0,0,0,0,0};
  uint8_t byteIdx  = absBit / 8;
  uint8_t bitInByte = absBit % 8;
  if (byteIdx < 6) f[byteIdx] = (uint8_t)(1u << bitInByte);
  char lbl[24];
  snprintf(lbl, sizeof(lbl), "TEST bit%u", absBit);
  do_send_frame(f, lbl);
}
static void test_send_zero() {
  uint8_t f[6] = {0,0,0,0,0,0};
  do_send_frame(f, "TEST gap");
}
static void test_start_common(bool once, uint8_t idxA, uint8_t idxB,
                               uint32_t onms, uint32_t offms) {
  preempt_runners();
  if (frameDirty) rebuild_staged_frame();
  memcpy(testWalker.restoreFrame, stagedFrame, 6);
  testWalker.active    = true;
  testWalker.oneShot   = once;
  testWalker.idxStart  = idxA;
  testWalker.idxEnd    = idxB;
  testWalker.idx       = idxA;
  testWalker.on_ms     = onms;
  testWalker.off_ms    = offms;
  testWalker.onPhase   = true;
  testWalker.phaseTimer = 0;
  test_send_one_hot(kTestBits[idxA]);
}

// ============================================================
// =================== COMMAND PARSER =========================
// ============================================================
static char cmdBuf[MAX_CMD_BUF];
static int  cmdLen = 0;

static void handle_command(char* rawLine) {
  // --- blank lines ---
  while (*rawLine == ' ' || *rawLine == '\t') ++rawLine;
  if (!*rawLine) return;

  // --- cmd tag @N ---
  g_tag[0] = '\0';
  if (rawLine[0] == '@') {
    char* sp = strchr(rawLine, ' ');
    if (sp) {
      int tlen = (int)(sp - rawLine);
      if (tlen < (int)sizeof(g_tag)) {
        memcpy(g_tag, rawLine, tlen);
        g_tag[tlen] = '\0';
      }
      rawLine = sp + 1;
    }
  }

  // Uppercase in-place
  str_upper(rawLine);

  // Tokenize (strtok mutates the buffer — that's fine)
  char* tok[MAX_TOKENS];
  int ntok = 0;
  char* t = strtok(rawLine, " \t");
  while (t && ntok < MAX_TOKENS) { tok[ntok++] = t; t = strtok(nullptr, " \t"); }
  if (ntok == 0) return;

  const char* cmd = tok[0];

  // ----------------------------------------------------------------
  // HELP
  // ----------------------------------------------------------------
  if (strcmp(cmd, "HELP") == 0) {
    Serial.println(F(
      "Base: ON OFF TOGGLE SEND PRINT/STATE CLEAR RESET FRAME\n"
      "  Names: OLF1_0..11  OLF2_0..11  SV1_0..1  SV2_0..1\n"
      "Macros (auto-SEND): CTRL OD1-5 <OLF1|OLF2|ALL>\n"
      "  CLN ODR <SV1|SV2|ALL> [ms]\n"
      "Pulse:  PULSE <names...> <ms>   PULSE?\n"
      "Seq:    SEQ NEW/ADD/END/RUN/STOP/LIST/DEL\n"
      "Preset: PRESET SAVE/LOAD/LIST/DEL <name>\n"
      "Test:   TEST START/ONCE/RANGE/STEP/STOP [params]\n"
      "Config: HB ON|OFF|<ms>  WATCH ON|OFF  SPD <hz>  STATUS\n"
      "Tag:    @N <cmd>  ->  all reply lines prefixed @N"
    ));
    return;
  }

  // ----------------------------------------------------------------
  // STATUS
  // ----------------------------------------------------------------
  if (strcmp(cmd, "STATUS") == 0) {
    if (frameDirty) rebuild_staged_frame();
    tag_print(); Serial.print("STATE bits: ");   print_state_line(stagedFrame);
    tag_print(); Serial.print("STATE frame: ");  print_frame_hex(stagedFrame);
    tag_print(); Serial.print("SPI hz: ");       Serial.println(g_spi_hz);
    tag_print(); Serial.print("HB: ");
    Serial.print(heartbeatOn ? "ON" : "OFF"); Serial.print(" @ "); Serial.print(heartbeatInterval); Serial.println(" ms");
    tag_print(); Serial.print("WATCH: "); Serial.println(watchOn ? "ON" : "OFF");
    tag_print(); Serial.print("TEST: ");
    if (testWalker.active)
      { Serial.print("ACTIVE idx="); Serial.print(testWalker.idx);
        Serial.print("/"); Serial.print(testWalker.idxEnd);
        Serial.print(" on="); Serial.print(testWalker.on_ms);
        Serial.print(" off="); Serial.println(testWalker.off_ms); }
    else Serial.println("idle");
    tag_print(); Serial.print("SEQ: ");
    if (seqRunner.active)
      { Serial.print("RUNNING "); Serial.print(sequences[seqRunner.seqIdx].name);
        Serial.print(" step="); Serial.print(seqRunner.stepIdx);
        Serial.print("/"); Serial.print(sequences[seqRunner.seqIdx].nsteps);
        Serial.print(" remain="); Serial.println(seqRunner.remaining); }
    else Serial.println("idle");
    // Pulse slots
    int activePulses = 0;
    for (int i = 0; i < MAX_PULSE_SLOTS; ++i) if (pulseSlots[i].active) activePulses++;
    tag_print(); Serial.print("PULSE slots active: "); Serial.println(activePulses);
    for (int i = 0; i < MAX_PULSE_SLOTS; ++i) {
      if (!pulseSlots[i].active) continue;
      int32_t rem = (int32_t)(pulseSlots[i].due_ms - millis());
      Serial.print("  ["); Serial.print(i); Serial.print("] due_in="); Serial.print(rem); Serial.println(" ms");
    }
    // Presets
    tag_print(); Serial.println("PRESETS:");
    for (int i = 0; i < MAX_PRESETS; ++i)
      if (presets[i].used) { Serial.print("  "); Serial.println(presets[i].name); }
    // Sequences
    tag_print(); Serial.println("SEQUENCES:");
    for (int i = 0; i < MAX_SEQ; ++i)
      if (sequences[i].defined) {
        Serial.print("  "); Serial.print(sequences[i].name);
        Serial.print(" ("); Serial.print(sequences[i].nsteps); Serial.println(" steps)");
      }
    return;
  }

  // ----------------------------------------------------------------
  // PRINT / STATE
  // ----------------------------------------------------------------
  if (strcmp(cmd, "PRINT") == 0 || strcmp(cmd, "STATE") == 0) {
    if (frameDirty) rebuild_staged_frame();
    tag_print(); Serial.print("STATE: "); print_state_line(stagedFrame);
    return;
  }

  // ----------------------------------------------------------------
  // CLEAR / RESET / SEND
  // ----------------------------------------------------------------
  if (strcmp(cmd, "CLEAR") == 0) {
    preempt_runners();
    staged = {0,0,0,0}; mark_dirty();
    tag_print(); Serial.println("CLEAR OK");
    return;
  }
  if (strcmp(cmd, "RESET") == 0) {
    preempt_runners(); cancel_all_pulses();
    staged = {0,0,0,0}; mark_dirty();
    do_send_staged();
    tag_print(); Serial.println("RESET OK");
    return;
  }
  if (strcmp(cmd, "SEND") == 0) {
    do_send_staged();
    return;
  }

  // ----------------------------------------------------------------
  // FRAME <h0> <h1> <h2> <h3> <h4> <h5>
  // ----------------------------------------------------------------
  if (strcmp(cmd, "FRAME") == 0) {
    if (ntok < 7) { tag_print(); Serial.println("ERR: FRAME needs 6 hex bytes"); return; }
    preempt_runners();
    uint8_t f[6];
    for (int i = 0; i < 6; ++i) f[i] = (uint8_t)strtoul(tok[i+1], nullptr, 16);
    memcpy(stagedFrame, f, 6);
    frame_to_bits(f, staged);
    frameDirty = false;
    do_send_frame(f, "FRAME");
    return;
  }

  // ----------------------------------------------------------------
  // ON / OFF / TOGGLE
  // ----------------------------------------------------------------
  if (strcmp(cmd,"ON")==0 || strcmp(cmd,"OFF")==0 || strcmp(cmd,"TOGGLE")==0) {
    int op = (cmd[1]=='N') ? +1 : (cmd[1]=='F') ? -1 : 0;
    if (ntok < 2) { tag_print(); Serial.println("ERR: no targets"); return; }
    bool ok = true;
    for (int i = 1; i < ntok; ++i)
      if (!set_bit(staged, tok[i], op)) {
        tag_print(); Serial.print("ERR: bad name "); Serial.println(tok[i]);
        ok = false;
      }
    mark_dirty();
    if (ok) { tag_print(); Serial.print(cmd); Serial.println(" OK (staged; use SEND to transmit)"); }
    return;
  }

  // ----------------------------------------------------------------
  // HB
  // ----------------------------------------------------------------
  if (strcmp(cmd, "HB") == 0) {
    if (ntok < 2) { tag_print(); Serial.println("ERR: HB needs ON|OFF|<ms>"); return; }
    if (strcmp(tok[1], "ON") == 0)  { heartbeatOn = true;  tag_print(); Serial.println("HB ON"); }
    else if (strcmp(tok[1], "OFF") == 0) { heartbeatOn = false; tag_print(); Serial.println("HB OFF"); }
    else if (is_numeric(tok[1])) {
      long v = atol(tok[1]);
      if (v <= 0) { tag_print(); Serial.println("ERR: HB ms must be >0"); return; }
      heartbeatInterval = (uint32_t)v; heartbeatOn = true;
      tag_print(); Serial.print("HB "); Serial.print(v); Serial.println(" ms (enabled)");
    } else { tag_print(); Serial.println("ERR: HB expects ON|OFF|<ms>"); }
    return;
  }

  // ----------------------------------------------------------------
  // WATCH
  // ----------------------------------------------------------------
  if (strcmp(cmd, "WATCH") == 0) {
    if (ntok < 2) { tag_print(); Serial.println("ERR: WATCH needs ON|OFF"); return; }
    if (strcmp(tok[1], "ON") == 0)  { watchOn = true;  tag_print(); Serial.println("WATCH ON"); }
    else if (strcmp(tok[1], "OFF") == 0) { watchOn = false; tag_print(); Serial.println("WATCH OFF"); }
    else { tag_print(); Serial.println("ERR: WATCH needs ON|OFF"); }
    return;
  }

  // ----------------------------------------------------------------
  // SPD <hz>
  // ----------------------------------------------------------------
  if (strcmp(cmd, "SPD") == 0) {
    if (ntok < 2 || !is_numeric(tok[1])) { tag_print(); Serial.println("ERR: SPD <hz>"); return; }
    g_spi_hz = (uint32_t)atol(tok[1]);
    tag_print(); Serial.print("SPD "); Serial.print(g_spi_hz); Serial.println(" Hz OK");
    return;
  }

  // ----------------------------------------------------------------
  // CLN / ODR  (switch macros, always before OD checks)
  // ----------------------------------------------------------------
  if (strcmp(cmd, "CLN") == 0 || strcmp(cmd, "ODR") == 0) {
    if (ntok < 2) { tag_print(); Serial.println("ERR: CLN/ODR needs SV1|SV2|ALL"); return; }
    bool toClean = (cmd[0] == 'C');
    bool s1=false, s2=false;
    if (!parse_sv_target(tok[1], s1, s2)) { tag_print(); Serial.println("ERR: bad switch target"); return; }
    preempt_runners();
    if (s1) staged.sv1 = toClean ? 0x00 : 0x03;
    if (s2) staged.sv2 = toClean ? 0x00 : 0x03;
    mark_dirty(); do_send_staged();
    long ms = (ntok >= 3 && is_numeric(tok[2])) ? atol(tok[2]) : -1;
    if (ms > 0) {
      // Schedule the flip-back using a pulse slot.
      // The slot's restoreFrame holds the state we want to APPLY at expiry
      // (opposite of what we just sent), then staged is updated to match.
      int slot = find_free_pulse();
      if (slot >= 0) {
        BitState flipped = staged;
        if (s1) flipped.sv1 = toClean ? 0x03 : 0x00;  // ODR after CLN, or CLN after ODR
        if (s2) flipped.sv2 = toClean ? 0x03 : 0x00;
        bits_to_frame(flipped, pulseSlots[slot].restoreFrame);
        pulseSlots[slot].active = true;
        pulseSlots[slot].due_ms = millis() + (uint32_t)ms;
      }
      tag_print();
      Serial.print(cmd); Serial.print(' '); Serial.print(tok[1]);
      Serial.print(" OK (flip in "); Serial.print(ms); Serial.println(" ms)");
    } else {
      tag_print(); Serial.print(cmd); Serial.print(' '); Serial.print(tok[1]); Serial.println(" OK");
    }
    return;
  }

  // ----------------------------------------------------------------
  // CTRL / OD1..OD5
  // ----------------------------------------------------------------
  {
    uint8_t firstBit;
    if (parse_olf_cmd(cmd, firstBit)) {
      if (ntok < 2) { tag_print(); Serial.println("ERR: needs OLF1|OLF2|ALL"); return; }
      bool o1=false, o2=false;
      if (!parse_olf_target(tok[1], o1, o2)) { tag_print(); Serial.println("ERR: bad OLF target"); return; }
      preempt_runners();
      olf_set_pair(staged, o1, o2, firstBit);
      mark_dirty(); do_send_staged();
      tag_print(); Serial.print(cmd); Serial.print(' '); Serial.print(tok[1]); Serial.println(" OK");
      return;
    }
  }

  // ----------------------------------------------------------------
  // PULSE <names...> <ms>    or    PULSE?
  // ----------------------------------------------------------------
  if (strcmp(cmd, "PULSE") == 0 || strcmp(cmd, "PULSE?") == 0) {
    if (strcmp(cmd, "PULSE?") == 0 || (ntok >= 2 && strcmp(tok[1], "?") == 0)) {
      int n = 0;
      for (int i = 0; i < MAX_PULSE_SLOTS; ++i) if (pulseSlots[i].active) ++n;
      tag_print(); Serial.print("PULSE active="); Serial.println(n);
      for (int i = 0; i < MAX_PULSE_SLOTS; ++i) {
        if (!pulseSlots[i].active) continue;
        int32_t rem = (int32_t)(pulseSlots[i].due_ms - millis());
        Serial.print("  ["); Serial.print(i); Serial.print("] "); Serial.print(rem); Serial.println(" ms remaining");
      }
      return;
    }
    // Last token must be duration
    if (ntok < 3) { tag_print(); Serial.println("ERR: PULSE <names...> <ms>"); return; }
    if (!is_numeric(tok[ntok-1])) { tag_print(); Serial.println("ERR: last PULSE arg must be ms"); return; }
    long on_ms = atol(tok[ntok-1]);
    if (on_ms <= 0) { tag_print(); Serial.println("ERR: PULSE ms must be >0"); return; }
    int slot = find_free_pulse();
    if (slot < 0) { tag_print(); Serial.println("ERR: no free pulse slots"); return; }
    preempt_runners();
    if (frameDirty) rebuild_staged_frame();
    // Snapshot restore frame BEFORE applying bits
    memcpy(pulseSlots[slot].restoreFrame, stagedFrame, 6);
    bool ok = true;
    for (int i = 1; i < ntok - 1; ++i)
      if (!set_bit(staged, tok[i], +1)) {
        tag_print(); Serial.print("ERR: bad name "); Serial.println(tok[i]);
        ok = false;
      }
    if (!ok) { pulseSlots[slot].active = false; frame_to_bits(pulseSlots[slot].restoreFrame, staged); mark_dirty(); return; }
    mark_dirty(); rebuild_staged_frame();
    do_send_frame(stagedFrame, "PULSE ON");
    pulseSlots[slot].active = true;
    pulseSlots[slot].due_ms = millis() + (uint32_t)on_ms;
    tag_print(); Serial.print("PULSE slot="); Serial.print(slot); Serial.print(" on_ms="); Serial.print(on_ms); Serial.println(" OK");
    return;
  }

  // ----------------------------------------------------------------
  // PRESET SAVE|LOAD|LIST|DEL <name>
  // ----------------------------------------------------------------
  if (strcmp(cmd, "PRESET") == 0) {
    if (ntok < 2) { tag_print(); Serial.println("ERR: PRESET SAVE|LOAD|LIST|DEL [name]"); return; }
    const char* sub = tok[1];
    if (strcmp(sub, "LIST") == 0) {
      tag_print(); Serial.println("PRESETS:");
      for (int i = 0; i < MAX_PRESETS; ++i)
        if (presets[i].used) { Serial.print("  "); Serial.println(presets[i].name); }
      return;
    }
    if (ntok < 3) { tag_print(); Serial.println("ERR: PRESET needs a name"); return; }
    const char* name = tok[2];
    if (strcmp(sub, "SAVE") == 0) {
      int slot = preset_find(name);
      if (slot < 0) slot = preset_free_slot();
      if (slot < 0) { tag_print(); Serial.println("ERR: preset storage full"); return; }
      if (frameDirty) rebuild_staged_frame();
      presets[slot].used = true;
      snprintf(presets[slot].name, MAX_PRESET_NAME, "%s", name);
      memcpy(presets[slot].frame, stagedFrame, 6);
      tag_print(); Serial.print("PRESET SAVE "); Serial.print(name); Serial.println(" OK");
    } else if (strcmp(sub, "LOAD") == 0) {
      int slot = preset_find(name);
      if (slot < 0) { tag_print(); Serial.print("ERR: preset not found: "); Serial.println(name); return; }
      preempt_runners();
      memcpy(stagedFrame, presets[slot].frame, 6);
      frame_to_bits(stagedFrame, staged);
      frameDirty = false;
      do_send_frame(stagedFrame, "PRESET LOAD");
      tag_print(); Serial.print("PRESET LOAD "); Serial.print(name); Serial.println(" OK");
    } else if (strcmp(sub, "DEL") == 0) {
      int slot = preset_find(name);
      if (slot < 0) { tag_print(); Serial.print("ERR: preset not found: "); Serial.println(name); return; }
      presets[slot].used = false;
      tag_print(); Serial.print("PRESET DEL "); Serial.print(name); Serial.println(" OK");
    } else {
      tag_print(); Serial.println("ERR: PRESET expects SAVE|LOAD|LIST|DEL");
    }
    return;
  }

  // ----------------------------------------------------------------
  // SEQ NEW|ADD|END|RUN|STOP|LIST|DEL
  // ----------------------------------------------------------------
  if (strcmp(cmd, "SEQ") == 0) {
    if (ntok < 2) { tag_print(); Serial.println("ERR: SEQ needs sub-command"); return; }
    const char* sub = tok[1];

    if (strcmp(sub, "NEW") == 0) {
      if (ntok < 3) { tag_print(); Serial.println("ERR: SEQ NEW <name>"); return; }
      if (defCursor.active) { tag_print(); Serial.println("ERR: already defining a sequence; SEQ END first"); return; }
      int slot = seq_find(tok[2]);
      if (slot >= 0) {
        // Redefine: clear it
        sequences[slot].nsteps = 0;
      } else {
        slot = seq_free_slot();
        if (slot < 0) { tag_print(); Serial.println("ERR: sequence storage full"); return; }
      }
      sequences[slot].defined = true;
      sequences[slot].nsteps  = 0;
      snprintf(sequences[slot].name, MAX_SEQ_NAME, "%s", tok[2]);
      defCursor.active = true;
      defCursor.seqIdx = (uint8_t)slot;
      defCursor.bits   = {0,0,0,0};
      tag_print(); Serial.print("SEQ NEW "); Serial.print(tok[2]); Serial.println(" — use SEQ ADD .. <ms>, SEQ END");
      return;
    }

    if (strcmp(sub, "ADD") == 0) {
      if (!defCursor.active) { tag_print(); Serial.println("ERR: no active SEQ NEW"); return; }
      Sequence& seq = sequences[defCursor.seqIdx];
      if (seq.nsteps >= MAX_SEQ_STEPS) { tag_print(); Serial.println("ERR: SEQ step limit reached"); return; }
      // Last token = duration
      if (ntok < 4) { tag_print(); Serial.println("ERR: SEQ ADD <cmd> [args...] <ms>"); return; }
      if (!is_numeric(tok[ntok-1])) { tag_print(); Serial.println("ERR: last token of SEQ ADD must be ms"); return; }
      uint32_t dur = (uint32_t)atol(tok[ntok-1]);
      // cmd tokens are tok[2..ntok-2]
      char** subTok = &tok[2];
      int subN      = ntok - 3;  // excludes SEQ, ADD, and the trailing ms
      if (!apply_seq_cmd(defCursor.bits, subTok, subN)) {
        tag_print(); Serial.println("ERR: SEQ ADD bad sub-command"); return;
      }
      uint8_t stepIdx = seq.nsteps++;
      bits_to_frame(defCursor.bits, seq.steps[stepIdx].frame);
      seq.steps[stepIdx].dur_ms = dur;
      tag_print(); Serial.print("SEQ ADD step "); Serial.print(stepIdx); Serial.print(" dur="); Serial.print(dur); Serial.println(" ms OK");
      return;
    }

    if (strcmp(sub, "END") == 0) {
      if (!defCursor.active) { tag_print(); Serial.println("ERR: no active SEQ NEW"); return; }
      defCursor.active = false;
      tag_print(); Serial.print("SEQ END — "); Serial.print(sequences[defCursor.seqIdx].nsteps); Serial.println(" steps saved");
      return;
    }

    if (strcmp(sub, "RUN") == 0) {
      if (ntok < 3) { tag_print(); Serial.println("ERR: SEQ RUN <name> [count]"); return; }
      int slot = seq_find(tok[2]);
      if (slot < 0) { tag_print(); Serial.print("ERR: sequence not found: "); Serial.println(tok[2]); return; }
      if (sequences[slot].nsteps == 0) { tag_print(); Serial.println("ERR: sequence is empty"); return; }
      preempt_runners();
      if (frameDirty) rebuild_staged_frame();
      memcpy(seqRunner.restoreFrame, stagedFrame, 6);
      seqRunner.active    = true;
      seqRunner.seqIdx    = (uint8_t)slot;
      seqRunner.stepIdx   = 0;
      seqRunner.stepSent  = false;
      seqRunner.remaining = (ntok >= 4) ? atol(tok[3]) : 1;
      seqRunner.stepTimer = 0;
      tag_print(); Serial.print("SEQ RUN "); Serial.print(tok[2]);
      Serial.print(" count="); Serial.println(seqRunner.remaining);
      return;
    }

    if (strcmp(sub, "STOP") == 0) {
      stop_seq();
      tag_print(); Serial.println("SEQ STOP OK");
      return;
    }

    if (strcmp(sub, "LIST") == 0) {
      tag_print(); Serial.println("SEQUENCES:");
      for (int i = 0; i < MAX_SEQ; ++i) {
        if (!sequences[i].defined) continue;
        Serial.print("  "); Serial.print(sequences[i].name);
        Serial.print("  ("); Serial.print(sequences[i].nsteps); Serial.println(" steps)");
      }
      return;
    }

    if (strcmp(sub, "DEL") == 0) {
      if (ntok < 3) { tag_print(); Serial.println("ERR: SEQ DEL <name>"); return; }
      int slot = seq_find(tok[2]);
      if (slot < 0) { tag_print(); Serial.print("ERR: sequence not found: "); Serial.println(tok[2]); return; }
      if (seqRunner.active && seqRunner.seqIdx == (uint8_t)slot) stop_seq();
      // Cancel definition cursor if this is the sequence being defined
      if (defCursor.active && defCursor.seqIdx == (uint8_t)slot) {
        defCursor.active = false;
        tag_print(); Serial.println("SEQ DEL: active definition cancelled");
      }
      sequences[slot].defined = false;
      tag_print(); Serial.print("SEQ DEL "); Serial.print(tok[2]); Serial.println(" OK");
      return;
    }

    tag_print(); Serial.println("ERR: SEQ expects NEW|ADD|END|RUN|STOP|LIST|DEL");
    return;
  }

  // ----------------------------------------------------------------
  // TEST START|ONCE|RANGE|STEP|STOP
  // ----------------------------------------------------------------
  if (strcmp(cmd, "TEST") == 0) {
    if (ntok < 2) { tag_print(); Serial.println("ERR: TEST START|ONCE|RANGE|STEP|STOP"); return; }
    const char* sub = tok[1];

    if (strcmp(sub, "START") == 0 || strcmp(sub, "ONCE") == 0) {
      uint32_t onms  = (ntok >= 3 && is_numeric(tok[2])) ? (uint32_t)atol(tok[2]) : 500;
      uint32_t offms = (ntok >= 4 && is_numeric(tok[3])) ? (uint32_t)atol(tok[3]) : 500;
      test_start_common(strcmp(sub,"ONCE")==0, 0, kTestBitCount-1, onms, offms);
      tag_print(); Serial.print("TEST "); Serial.print(sub);
      Serial.print(" on="); Serial.print(onms); Serial.print(" ms off="); Serial.print(offms); Serial.println(" ms");
      return;
    }

    if (strcmp(sub, "RANGE") == 0) {
      if (ntok < 4) { tag_print(); Serial.println("ERR: TEST RANGE <a> <b> [on_ms [off_ms]]"); return; }
      uint8_t a = (uint8_t)atoi(tok[2]);
      uint8_t b = (uint8_t)atoi(tok[3]);
      if (a >= kTestBitCount) a = 0;
      if (b >= kTestBitCount) b = kTestBitCount - 1;
      if (a > b) { uint8_t tmp=a; a=b; b=tmp; }
      uint32_t onms  = (ntok >= 5 && is_numeric(tok[4])) ? (uint32_t)atol(tok[4]) : 500;
      uint32_t offms = (ntok >= 6 && is_numeric(tok[5])) ? (uint32_t)atol(tok[5]) : 500;
      test_start_common(false, a, b, onms, offms);
      tag_print(); Serial.print("TEST RANGE "); Serial.print(a); Serial.print(".."); Serial.print(b);
      Serial.print(" on="); Serial.print(onms); Serial.print(" off="); Serial.println(offms);
      return;
    }

    if (strcmp(sub, "STEP") == 0) {
      // Manual advance one bit (useful for diagnostic stepping without timing pressure)
      if (!testWalker.active) {
        // Start in manual mode: cue first bit then pause
        test_start_common(true, 0, kTestBitCount-1, 0xFFFFFFFF, 0xFFFFFFFF);
        tag_print(); Serial.println("TEST STEP — manual mode started");
      } else {
        testWalker.idx = (testWalker.idx < testWalker.idxEnd) ? testWalker.idx+1 : testWalker.idxStart;
        testWalker.onPhase = true;
        testWalker.phaseTimer = 0;
        test_send_one_hot(kTestBits[testWalker.idx]);
        tag_print(); Serial.print("TEST STEP bit="); Serial.println(kTestBits[testWalker.idx]);
      }
      return;
    }

    if (strcmp(sub, "STOP") == 0) {
      stop_test();
      tag_print(); Serial.println("TEST STOP OK");
      return;
    }

    tag_print(); Serial.println("ERR: TEST expects START|ONCE|RANGE|STEP|STOP");
    return;
  }

  // ----------------------------------------------------------------
  // Unknown
  // ----------------------------------------------------------------
  tag_print(); Serial.print("ERR: unknown command '"); Serial.print(cmd); Serial.println("' — type HELP");
}

// ============================================================
// =================== BINARY PROTOCOL ========================
// ============================================================
// Reads a binary packet after the 0xAA sentinel has been consumed.
// Packet: [cmd:1][len:1][payload:len][crc8:1]
// CRC-8 covers cmd + payload (NOT the len byte).
static void handle_binary(uint8_t firstByte) {
  // Wait for len byte (up to 5 ms)
  uint32_t t0 = millis();
  uint8_t pkt[2 + MAX_BIN_PAYLOAD]; // cmd + payload + crc (max)
  pkt[0] = firstByte;   // cmd
  // Read len
  while (!Serial.available() && millis()-t0 < 5);
  if (!Serial.available()) { Serial.println("BIN ERR: timeout"); return; }
  uint8_t len = (uint8_t)Serial.read();
  if (len > MAX_BIN_PAYLOAD) {
    Serial.println("BIN ERR: payload too large");
    // Drain remaining bytes to re-sync
    t0 = millis();
    while (millis()-t0 < 15) { if (Serial.available()) Serial.read(); }
    return;
  }
  // Read payload + crc
  uint8_t total = len + 1;  // payload + 1 crc byte
  uint8_t got = 0;
  t0 = millis();
  while (got < total) {
    if (Serial.available()) pkt[1 + got++] = (uint8_t)Serial.read();
    else if (millis()-t0 > 10) break;
  }
  if (got < total) { Serial.println("BIN ERR: short packet"); return; }
  // Check CRC (covers cmd + payload, NOT the len byte)
  uint8_t crcCalc = crc8_buf(pkt, 1 + len);
  uint8_t crcRecv = pkt[1 + len];
  if (crcCalc != crcRecv) { Serial.println("BIN ERR: bad CRC"); return; }

  uint8_t cmd  = pkt[0];
  uint8_t* pay = &pkt[1];

  switch (cmd) {
    case 0x01: // SEND_FRAME
      if (len < 6) { Serial.write(0xAB); Serial.write(0x01); return; }
      memcpy(stagedFrame, pay, 6);
      frame_to_bits(stagedFrame, staged); frameDirty = false;
      spi_send_raw(stagedFrame);
      Serial.write(0xAB); Serial.write((uint8_t)0x00);
      break;
    case 0x02: { // PULSE_FRAME
      if (len < 10) { Serial.write(0xAB); Serial.write(0x02); return; }
      uint32_t ms; memcpy(&ms, pay+6, 4);
      int slot = find_free_pulse();
      if (slot < 0) { Serial.write(0xAB); Serial.write(0x10); return; }
      if (frameDirty) rebuild_staged_frame();
      memcpy(pulseSlots[slot].restoreFrame, stagedFrame, 6);
      memcpy(stagedFrame, pay, 6);
      frame_to_bits(stagedFrame, staged); frameDirty = false;
      spi_send_raw(stagedFrame);
      pulseSlots[slot].active = true;
      pulseSlots[slot].due_ms = millis() + ms;
      Serial.write(0xAB); Serial.write((uint8_t)0x00);
      break;
    }
    case 0x03: { // QUERY
      if (frameDirty) rebuild_staged_frame();
      Serial.write(0xAB); Serial.write((uint8_t)0x03);
      for (int i = 0; i < 6; ++i) Serial.write(stagedFrame[i]);
      break;
    }
    case 0x04: { // SEQ_RUN
      char nm[MAX_SEQ_NAME]; int n=0;
      while (n < len && n < MAX_SEQ_NAME-1) nm[n] = (char)pay[n], ++n;
      nm[n] = '\0'; str_upper(nm);
      int slot = seq_find(nm);
      if (slot < 0) { Serial.write(0xAB); Serial.write(0x20); return; }
      preempt_runners();
      if (frameDirty) rebuild_staged_frame();
      memcpy(seqRunner.restoreFrame, stagedFrame, 6);
      seqRunner.active=true; seqRunner.seqIdx=(uint8_t)slot;
      seqRunner.stepIdx=0; seqRunner.stepSent=false;
      seqRunner.remaining=1; seqRunner.stepTimer=0;
      Serial.write(0xAB); Serial.write((uint8_t)0x00);
      break;
    }
    case 0x05: // SEQ_STOP
      stop_seq();
      Serial.write(0xAB); Serial.write((uint8_t)0x00);
      break;
    case 0x06: { // PRESET_LOAD
      char nm[MAX_PRESET_NAME]; int n=0;
      while (n < len && n < MAX_PRESET_NAME-1) nm[n]=(char)pay[n], ++n;
      nm[n]='\0'; str_upper(nm);
      int slot=preset_find(nm);
      if (slot<0) { Serial.write(0xAB); Serial.write(0x21); return; }
      preempt_runners();
      memcpy(stagedFrame,presets[slot].frame,6);
      frame_to_bits(stagedFrame,staged); frameDirty=false;
      spi_send_raw(stagedFrame);
      Serial.write(0xAB); Serial.write((uint8_t)0x00);
      break;
    }
    case 0x07: { // PRESET_SAVE
      char nm[MAX_PRESET_NAME]; int n=0;
      while (n < len && n < MAX_PRESET_NAME-1) nm[n]=(char)pay[n], ++n;
      nm[n]='\0'; str_upper(nm);
      int slot=preset_find(nm); if(slot<0) slot=preset_free_slot();
      if(slot<0){ Serial.write(0xAB); Serial.write(0x22); return; }
      if(frameDirty) rebuild_staged_frame();
      presets[slot].used=true;
      snprintf(presets[slot].name, MAX_PRESET_NAME, "%s", nm);
      memcpy(presets[slot].frame,stagedFrame,6);
      Serial.write(0xAB); Serial.write((uint8_t)0x00);
      break;
    }
    default:
      Serial.write(0xAB); Serial.write(0xFF);
      break;
  }
}

// ============================================================
// ======================== SETUP / LOOP ======================
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("=== TPIC6B595 controller v2 (Teensy 4.1) ===");
  Serial.println("Type HELP for commands. HB ON to enable heartbeat.");
  pinMode(PIN_CS, OUTPUT);
  SPI.begin();
  memset(lastSentFrame, 0, 6);
  heartbeatTimer = 0;
}

void loop() {
  // ---- serial line reader (text) + binary sentinel ----
  while (Serial.available()) {
    uint8_t c = (uint8_t)Serial.read();
    if (c == 0xAA && cmdLen == 0) {
      uint8_t cmd_byte;
      uint32_t t0 = millis();
      while (!Serial.available() && millis()-t0 < 5);
      if (!Serial.available()) break;
      cmd_byte = (uint8_t)Serial.read();
      handle_binary(cmd_byte);
      continue;
    }
    if (c == '\r') continue;
    if (c == '\n') {
      cmdBuf[cmdLen] = '\0';
      handle_command(cmdBuf);
      cmdLen = 0;
    } else {
      if (cmdLen < MAX_CMD_BUF - 1) cmdBuf[cmdLen++] = (char)c;
    }
  }

  // ---- pulse timer expiry ----
  uint32_t now = millis();
  for (int i = 0; i < MAX_PULSE_SLOTS; ++i) {
    if (pulseSlots[i].active && (int32_t)(now - pulseSlots[i].due_ms) >= 0) {
      pulseSlots[i].active = false;
      // Restore frame
      memcpy(stagedFrame, pulseSlots[i].restoreFrame, 6);
      frame_to_bits(stagedFrame, staged);
      frameDirty = false;
      do_send_frame(stagedFrame, "PULSE OFF restore");
    }
  }

  // ---- sequence runner ----
  if (seqRunner.active) {
    Sequence& seq = sequences[seqRunner.seqIdx];
    if (!seqRunner.stepSent) {
      do_send_frame(seq.steps[seqRunner.stepIdx].frame, "SEQ step");
      seqRunner.stepSent  = true;
      seqRunner.stepTimer = 0;
    } else if (seqRunner.stepTimer >= seq.steps[seqRunner.stepIdx].dur_ms) {
      ++seqRunner.stepIdx;
      if (seqRunner.stepIdx >= seq.nsteps) {
        // End of pass
        if (seqRunner.remaining > 0) --seqRunner.remaining;
        if (seqRunner.remaining == 0) {
          seqRunner.active = false;
          do_send_frame(seqRunner.restoreFrame, "SEQ done restore");
          memcpy(stagedFrame, seqRunner.restoreFrame, 6);
          frame_to_bits(stagedFrame, staged); frameDirty = false;
        } else {
          seqRunner.stepIdx = 0;  // loop
        }
      }
      seqRunner.stepSent = false;
    }
  }

  // ---- test walker ----
  if (testWalker.active) {
    if (testWalker.onPhase) {
      if (testWalker.phaseTimer >= testWalker.on_ms) {
        testWalker.onPhase = false;
        testWalker.phaseTimer = 0;
        test_send_zero();
      }
    } else {
      if (testWalker.phaseTimer >= testWalker.off_ms) {
        testWalker.onPhase = true;
        testWalker.phaseTimer = 0;
        if (testWalker.idx >= testWalker.idxEnd) {
          if (testWalker.oneShot) {
            stop_test();
            Serial.println("TEST ONCE complete");
          } else {
            testWalker.idx = testWalker.idxStart;
            test_send_one_hot(kTestBits[testWalker.idx]);
          }
        } else {
          ++testWalker.idx;
          test_send_one_hot(kTestBits[testWalker.idx]);
        }
      }
    }
  }

  // ---- heartbeat ----
  if (heartbeatOn && heartbeatTimer >= heartbeatInterval) {
    heartbeatTimer = 0;
    if (frameDirty) rebuild_staged_frame();
    Serial.print("HB: ");
    print_state_line(stagedFrame);
  }
}
