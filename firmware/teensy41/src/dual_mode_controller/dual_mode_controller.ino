#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#ifndef BIT
#define BIT(n) (1u << (n))
#endif

// -------- Shared SPI --------
constexpr uint32_t SPI_HZ = 1'000'000;
constexpr int PIN_CS = 10;

// -------- Open-loop wiring --------
constexpr int PIN_S_OLF1_S1 = 0;
constexpr int PIN_S_SV1_S0  = 1;
constexpr int PIN_S_OLF2_S1 = 2;
constexpr int PIN_S_SV2_S0  = 3;
constexpr int PIN_S_OLF2_S2 = 4;
constexpr int PIN_S_OLF2_S0 = 5;
constexpr int PIN_S_OLF1_S2 = 6;
constexpr int PIN_S_OLF1_S0 = 7;

constexpr int PIN_GLOBAL_LOAD = 23;

constexpr int PIN_RCK_SENSE_OLF1 = 19;
constexpr int PIN_RCK_SENSE_OLF2 = 20;
constexpr int PIN_RCK_SENSE_SV2  = 21;
constexpr int PIN_RCK_SENSE_SV1  = 22;

constexpr int PIN_READY_OLF1 = 24;
constexpr int PIN_READY_SV1  = 25;
constexpr int PIN_READY_SV2  = 26;
constexpr int PIN_READY_OLF2 = 27;

// -------- Serial-mode command parsing --------
constexpr int MAX_CMD_BUF = 128;
constexpr int MAX_TOKENS = 12;

// -------- Shared frame map --------
enum FrameSlot : uint8_t {
  SLOT_OLF1_HI = 0,
  SLOT_OLF1_LO,
  SLOT_OLF2_HI,
  SLOT_OLF2_LO,
  SLOT_SV1,
  SLOT_SV2,
};

uint8_t FRAME_SEND_ORDER[6] = {
  SLOT_OLF1_LO,
  SLOT_OLF1_HI,
  SLOT_OLF2_LO,
  SLOT_OLF2_HI,
  SLOT_SV2,
  SLOT_SV1,
};

// -------- Minimal serial controller state --------
struct BitState {
  uint16_t olf1 = 0;
  uint16_t olf2 = 0;
  uint8_t sv1 = 0;
  uint8_t sv2 = 0;
};

BitState staged;
uint8_t stagedFrame[6] = {0, 0, 0, 0, 0, 0};
bool frameDirty = true;
uint8_t lastSentFrame[6] = {0, 0, 0, 0, 0, 0};

// -------- Open-loop logging --------
File logFile;
bool sdReady = false;
bool openLoopLogReady = false;
constexpr int LOG_BUFFER_SIZE = 64;
constexpr size_t LOG_LINE_MAX = 192;
volatile char logBuffer[LOG_BUFFER_SIZE][LOG_LINE_MAX];
volatile int logHead = 0;
volatile int logTail = 0;

// -------- Open-loop states --------
enum : uint8_t { ST_OFF = 0, ST_AIR, ST_ODOR1, ST_ODOR2, ST_ODOR3, ST_ODOR4, ST_ODOR5, ST_FLUSH };

constexpr uint16_t OLFACTOMETER_STATES[8] = {
  0x0000,
  BIT(0) | BIT(1),
  BIT(2) | BIT(3),
  BIT(4) | BIT(5),
  BIT(6) | BIT(7),
  BIT(8) | BIT(9),
  BIT(10) | BIT(11),
  (uint16_t)0x0FFF,
};

constexpr uint8_t SWITCH_STATES_2LVL[2] = {
  0b00000000,
  0b00000011,
};

enum FirmwareMode : uint8_t {
  MODE_SERIAL = 0,
  MODE_OPEN_LOOP = 1,
};

volatile FirmwareMode currentMode = MODE_SERIAL;
bool openLoopPinsInitialized = false;
bool openLoopInterruptsAttached = false;

volatile bool ready_olf1 = false;
volatile bool ready_olf2 = false;
volatile bool ready_sv1 = false;
volatile bool ready_sv2 = false;

// -------- Serial-mode TEST walker --------
constexpr uint8_t kTestBits[] = {
  0,1,2,3,4,5,6,7,8,9,10,11,
  16,17,18,19,20,21,22,23,24,25,26,27,
  32,33,
  40,41
};
constexpr uint8_t kTestBitCount = sizeof(kTestBits) / sizeof(kTestBits[0]);

struct TestWalker {
  bool active = false;
  bool onPhase = true;
  uint8_t idx = 0;
  uint32_t on_ms = 500;
  uint32_t off_ms = 500;
  elapsedMillis phaseTimer;
  uint8_t restoreFrame[6] = {0, 0, 0, 0, 0, 0};
};

TestWalker testWalker;
char cmdBuf[MAX_CMD_BUF];
int cmdLen = 0;

static inline void enqueue_logf(const char* fmt, ...) {
  int nextHead = (logHead + 1) % LOG_BUFFER_SIZE;
  if (nextHead == logTail) {
    return;
  }

  va_list args;
  va_start(args, fmt);
  vsnprintf((char*)logBuffer[logHead], LOG_LINE_MAX, fmt, args);
  va_end(args);
  logHead = nextHead;
}

static inline void bits_to_frame(const BitState& bits, uint8_t frame[6]) {
  frame[0] = (uint8_t)(bits.olf1 & 0xFF);
  frame[1] = (uint8_t)((bits.olf1 >> 8) & 0x0F);
  frame[2] = (uint8_t)(bits.olf2 & 0xFF);
  frame[3] = (uint8_t)((bits.olf2 >> 8) & 0x0F);
  frame[4] = (uint8_t)(bits.sv2 & 0x03);
  frame[5] = (uint8_t)(bits.sv1 & 0x03);
}

static inline void frame_to_bits(const uint8_t frame[6], BitState& bits) {
  bits.olf1 = frame[0] | ((uint16_t)(frame[1] & 0x0F) << 8);
  bits.olf2 = frame[2] | ((uint16_t)(frame[3] & 0x0F) << 8);
  bits.sv2 = frame[4] & 0x03;
  bits.sv1 = frame[5] & 0x03;
}

static inline void rebuild_staged_frame() {
  bits_to_frame(staged, stagedFrame);
  frameDirty = false;
}

static inline void mark_dirty() {
  frameDirty = true;
}

static inline void spi_send_raw(const uint8_t frame[6]) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  for (int i = 5; i >= 0; --i) {
    SPI.transfer(frame[i]);
  }
  SPI.endTransaction();
}

static void print_frame_hex(const uint8_t frame[6]) {
  for (int i = 5; i >= 0; --i) {
    if (frame[i] < 0x10) {
      Serial.print('0');
    }
    Serial.print(frame[i], HEX);
    if (i > 0) {
      Serial.print(' ');
    }
  }
  Serial.println();
}

static void print_state_line(const uint8_t frame[6]) {
  BitState bits;
  frame_to_bits(frame, bits);
  char buf[34];
  int pos = 0;
  for (int i = 11; i >= 0; --i) buf[pos++] = ((bits.olf1 >> i) & 1) ? '1' : '0';
  buf[pos++] = ' ';
  for (int i = 11; i >= 0; --i) buf[pos++] = ((bits.olf2 >> i) & 1) ? '1' : '0';
  buf[pos++] = ' ';
  for (int i = 1; i >= 0; --i) buf[pos++] = ((bits.sv1 >> i) & 1) ? '1' : '0';
  buf[pos++] = ' ';
  for (int i = 1; i >= 0; --i) buf[pos++] = ((bits.sv2 >> i) & 1) ? '1' : '0';
  buf[pos] = '\0';
  Serial.println(buf);
}

static void do_send_frame(const uint8_t frame[6], const char* label) {
  spi_send_raw(frame);
  Serial.print(label);
  Serial.print(" Data: ");
  print_frame_hex(frame);
  memcpy(lastSentFrame, frame, 6);
}

static void do_send_staged() {
  if (frameDirty) {
    rebuild_staged_frame();
  }
  do_send_frame(stagedFrame, "SEND");
}

static bool is_numeric(const char* text) {
  if (!text || !*text) {
    return false;
  }
  const char* cursor = text;
  if (*cursor == '-') {
    ++cursor;
  }
  if (!*cursor) {
    return false;
  }
  for (; *cursor; ++cursor) {
    if (*cursor < '0' || *cursor > '9') {
      return false;
    }
  }
  return true;
}

static void str_upper(char* text) {
  for (; *text; ++text) {
    if (*text >= 'a' && *text <= 'z') {
      *text -= 32;
    }
  }
}

static bool parse_olf_cmd(const char* cmd, uint8_t& firstBit) {
  if (strcmp(cmd, "CTRL") == 0) {
    firstBit = 0;
    return true;
  }
  if (strlen(cmd) == 3 && cmd[0] == 'O' && cmd[1] == 'D' && cmd[2] >= '1' && cmd[2] <= '5') {
    firstBit = (uint8_t)(2 * (cmd[2] - '0'));
    return true;
  }
  return false;
}

static bool parse_olf_target(const char* text, bool& olf1, bool& olf2) {
  if (strcmp(text, "OLF1") == 0) {
    olf1 = true;
    olf2 = false;
    return true;
  }
  if (strcmp(text, "OLF2") == 0) {
    olf1 = false;
    olf2 = true;
    return true;
  }
  if (strcmp(text, "ALL") == 0) {
    olf1 = true;
    olf2 = true;
    return true;
  }
  return false;
}

static bool parse_sv_target(const char* text, bool& sv1, bool& sv2) {
  if (strcmp(text, "SV1") == 0) {
    sv1 = true;
    sv2 = false;
    return true;
  }
  if (strcmp(text, "SV2") == 0) {
    sv1 = false;
    sv2 = true;
    return true;
  }
  if (strcmp(text, "ALL") == 0) {
    sv1 = true;
    sv2 = true;
    return true;
  }
  return false;
}

static void olf_set_pair(BitState& bits, bool setOlf1, bool setOlf2, uint8_t firstBit) {
  uint16_t value = 0;
  if (firstBit < 12) value |= (1u << firstBit);
  if (firstBit + 1 < 12) value |= (1u << (firstBit + 1));
  if (setOlf1) bits.olf1 = value;
  if (setOlf2) bits.olf2 = value;
}

static void stop_test(bool announce) {
  if (!testWalker.active) {
    return;
  }
  testWalker.active = false;
  do_send_frame(testWalker.restoreFrame, "TEST STOP restore");
  memcpy(stagedFrame, testWalker.restoreFrame, 6);
  frame_to_bits(testWalker.restoreFrame, staged);
  frameDirty = false;
  if (announce) {
    Serial.println("TEST STOP OK");
  }
}

static void preempt_test() {
  if (testWalker.active) {
    stop_test(false);
    Serial.println("TEST preempted");
  }
}

static void test_send_one_hot(uint8_t absBit) {
  uint8_t frame[6] = {0, 0, 0, 0, 0, 0};
  uint8_t byteIdx = absBit / 8;
  uint8_t bitInByte = absBit % 8;
  if (byteIdx < 6) {
    frame[byteIdx] = (uint8_t)(1u << bitInByte);
  }
  char label[24];
  snprintf(label, sizeof(label), "TEST bit%u", absBit);
  do_send_frame(frame, label);
}

static void test_send_zero() {
  uint8_t frame[6] = {0, 0, 0, 0, 0, 0};
  do_send_frame(frame, "TEST gap");
}

static void start_test(uint32_t onMs, uint32_t offMs) {
  preempt_test();
  if (frameDirty) {
    rebuild_staged_frame();
  }
  memcpy(testWalker.restoreFrame, stagedFrame, 6);
  testWalker.active = true;
  testWalker.onPhase = true;
  testWalker.idx = 0;
  testWalker.on_ms = onMs;
  testWalker.off_ms = offMs;
  testWalker.phaseTimer = 0;
  test_send_one_hot(kTestBits[0]);
}

static void init_open_loop_pins() {
  if (openLoopPinsInitialized) {
    return;
  }

  pinMode(PIN_READY_OLF1, OUTPUT); digitalWriteFast(PIN_READY_OLF1, LOW);
  pinMode(PIN_READY_SV1, OUTPUT);  digitalWriteFast(PIN_READY_SV1, LOW);
  pinMode(PIN_READY_SV2, OUTPUT);  digitalWriteFast(PIN_READY_SV2, LOW);
  pinMode(PIN_READY_OLF2, OUTPUT); digitalWriteFast(PIN_READY_OLF2, LOW);

  pinMode(PIN_S_OLF1_S1, INPUT);
  pinMode(PIN_S_SV1_S0, INPUT);
  pinMode(PIN_S_OLF2_S1, INPUT);
  pinMode(PIN_S_SV2_S0, INPUT);
  pinMode(PIN_S_OLF2_S2, INPUT);
  pinMode(PIN_S_OLF2_S0, INPUT);
  pinMode(PIN_S_OLF1_S2, INPUT);
  pinMode(PIN_S_OLF1_S0, INPUT);

  pinMode(PIN_GLOBAL_LOAD, INPUT);
  pinMode(PIN_RCK_SENSE_OLF1, INPUT);
  pinMode(PIN_RCK_SENSE_OLF2, INPUT);
  pinMode(PIN_RCK_SENSE_SV2, INPUT);
  pinMode(PIN_RCK_SENSE_SV1, INPUT);

  openLoopPinsInitialized = true;
}

static bool init_open_loop_logging() {
  if (openLoopLogReady) {
    return true;
  }

  if (!sdReady) {
    if (!SD.begin(BUILTIN_SDCARD)) {
      Serial.println("FAULT code=SD_INIT_FAILED action=OPENLOOP_ABORT");
      return false;
    }
    sdReady = true;
    Serial.println("MODE sd_card=ready");
  }

  char logFileName[] = "log_000.txt";
  for (int i = 0; i < 1000; ++i) {
    logFileName[4] = i / 100 + '0';
    logFileName[5] = (i / 10) % 10 + '0';
    logFileName[6] = i % 10 + '0';
    if (!SD.exists(logFileName)) {
      break;
    }
  }

  logFile = SD.open(logFileName, FILE_WRITE);
  if (!logFile) {
    Serial.println("FAULT code=LOG_OPEN_FAILED action=OPENLOOP_ABORT");
    return false;
  }

  openLoopLogReady = true;
  Serial.print("MODE log_file=");
  Serial.println(logFileName);
  logFile.println("MODE name=open_loop_controller version=1 transport=usb_serial");
  logFile.flush();
  return true;
}

static inline void build_open_loop_frame(uint8_t out[6]) {
  uint8_t olf1_idx = ((digitalReadFast(PIN_S_OLF1_S2) & 1) << 2) |
                     ((digitalReadFast(PIN_S_OLF1_S1) & 1) << 1) |
                     ((digitalReadFast(PIN_S_OLF1_S0) & 1) << 0);
  uint8_t olf2_idx = ((digitalReadFast(PIN_S_OLF2_S2) & 1) << 2) |
                     ((digitalReadFast(PIN_S_OLF2_S1) & 1) << 1) |
                     ((digitalReadFast(PIN_S_OLF2_S0) & 1) << 0);
  uint8_t sv1_idx = (digitalReadFast(PIN_S_SV1_S0) & 1);
  uint8_t sv2_idx = (digitalReadFast(PIN_S_SV2_S0) & 1);

  uint16_t olf1_val = OLFACTOMETER_STATES[olf1_idx & 0x07];
  uint16_t olf2_val = OLFACTOMETER_STATES[olf2_idx & 0x07];
  uint8_t sv1_val = SWITCH_STATES_2LVL[sv1_idx & 0x01];
  uint8_t sv2_val = SWITCH_STATES_2LVL[sv2_idx & 0x01];

  uint8_t slots[6];
  slots[SLOT_OLF1_HI] = (uint8_t)((olf1_val >> 8) & 0xFF);
  slots[SLOT_OLF1_LO] = (uint8_t)(olf1_val & 0xFF);
  slots[SLOT_OLF2_HI] = (uint8_t)((olf2_val >> 8) & 0xFF);
  slots[SLOT_OLF2_LO] = (uint8_t)(olf2_val & 0xFF);
  slots[SLOT_SV1] = sv1_val;
  slots[SLOT_SV2] = sv2_val;

  for (int i = 0; i < 6; ++i) {
    out[i] = slots[FRAME_SEND_ORDER[i]];
  }

  unsigned long t = micros();
  enqueue_logf(
    "VALVE t_us=%lu olf1_state=%u olf2_state=%u sv1_state=%u sv2_state=%u olf1_bits=0x%04X olf2_bits=0x%04X sv1_bits=0x%02X sv2_bits=0x%02X",
    t, olf1_idx, olf2_idx, sv1_idx, sv2_idx, olf1_val, olf2_val, sv1_val, sv2_val
  );
}

static void attach_open_loop_interrupts();
static void detach_open_loop_interrupts();

void isr_global_load() {
  if (currentMode != MODE_OPEN_LOOP) {
    return;
  }
  uint8_t bytes[6];
  build_open_loop_frame(bytes);
  spi_send_raw(bytes);

  ready_olf1 = ready_olf2 = ready_sv1 = ready_sv2 = true;
  digitalWriteFast(PIN_READY_OLF1, HIGH);
  digitalWriteFast(PIN_READY_OLF2, HIGH);
  digitalWriteFast(PIN_READY_SV1, HIGH);
  digitalWriteFast(PIN_READY_SV2, HIGH);
  enqueue_logf("READY t_us=%lu target=ALL olf1=1 olf2=1 sv1=1 sv2=1 reason=LOAD", micros());
}

void isr_rck_olf1() {
  if (currentMode != MODE_OPEN_LOOP || !ready_olf1) {
    return;
  }
  ready_olf1 = false;
  digitalWriteFast(PIN_READY_OLF1, LOW);
  unsigned long t = micros();
  enqueue_logf("COMMIT t_us=%lu target=OLF1", t);
  enqueue_logf("READY t_us=%lu target=OLF1 value=0 reason=RCK", t);
}

void isr_rck_olf2() {
  if (currentMode != MODE_OPEN_LOOP || !ready_olf2) {
    return;
  }
  ready_olf2 = false;
  digitalWriteFast(PIN_READY_OLF2, LOW);
  unsigned long t = micros();
  enqueue_logf("COMMIT t_us=%lu target=OLF2", t);
  enqueue_logf("READY t_us=%lu target=OLF2 value=0 reason=RCK", t);
}

void isr_rck_sv2() {
  if (currentMode != MODE_OPEN_LOOP || !ready_sv2) {
    return;
  }
  ready_sv2 = false;
  digitalWriteFast(PIN_READY_SV2, LOW);
  unsigned long t = micros();
  enqueue_logf("COMMIT t_us=%lu target=SV2", t);
  enqueue_logf("READY t_us=%lu target=SV2 value=0 reason=RCK", t);
}

void isr_rck_sv1() {
  if (currentMode != MODE_OPEN_LOOP || !ready_sv1) {
    return;
  }
  ready_sv1 = false;
  digitalWriteFast(PIN_READY_SV1, LOW);
  unsigned long t = micros();
  enqueue_logf("COMMIT t_us=%lu target=SV1", t);
  enqueue_logf("READY t_us=%lu target=SV1 value=0 reason=RCK", t);
}

static void attach_open_loop_interrupts() {
  if (openLoopInterruptsAttached) {
    return;
  }
  attachInterrupt(digitalPinToInterrupt(PIN_GLOBAL_LOAD), isr_global_load, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLF1), isr_rck_olf1, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLF2), isr_rck_olf2, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SV2), isr_rck_sv2, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SV1), isr_rck_sv1, RISING);
  openLoopInterruptsAttached = true;
}

static void detach_open_loop_interrupts() {
  if (!openLoopInterruptsAttached) {
    return;
  }
  detachInterrupt(digitalPinToInterrupt(PIN_GLOBAL_LOAD));
  detachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLF1));
  detachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_OLF2));
  detachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SV2));
  detachInterrupt(digitalPinToInterrupt(PIN_RCK_SENSE_SV1));
  openLoopInterruptsAttached = false;
}

static void enter_open_loop_mode() {
  if (currentMode == MODE_OPEN_LOOP) {
    Serial.println("OPENLOOP already active");
    return;
  }

  preempt_test();
  init_open_loop_pins();
  if (!init_open_loop_logging()) {
    return;
  }

  ready_olf1 = ready_olf2 = ready_sv1 = ready_sv2 = false;
  digitalWriteFast(PIN_READY_OLF1, LOW);
  digitalWriteFast(PIN_READY_OLF2, LOW);
  digitalWriteFast(PIN_READY_SV1, LOW);
  digitalWriteFast(PIN_READY_SV2, LOW);

  currentMode = MODE_OPEN_LOOP;
  attach_open_loop_interrupts();

  uint8_t initBytes[6] = {0, 0, 0, 0, 0, 0};
  spi_send_raw(initBytes);

  Serial.println("MODE name=open_loop_controller version=1 transport=usb_serial");
  enqueue_logf("VALVE t_us=%lu olf1_state=0 olf2_state=0 sv1_state=0 sv2_state=0 reason=BOOT_INIT", micros());
  enqueue_logf("READY t_us=%lu target=ALL olf1=0 olf2=0 sv1=0 sv2=0 reason=BOOT", micros());
  enqueue_logf("MODE t_us=%lu state=OPEN_LOOP reason=CMD_START", micros());
  Serial.println("OPENLOOP START OK");
}

static void exit_open_loop_mode() {
  if (currentMode != MODE_OPEN_LOOP) {
    Serial.println("OPENLOOP already stopped");
    return;
  }

  detach_open_loop_interrupts();
  currentMode = MODE_SERIAL;

  ready_olf1 = ready_olf2 = ready_sv1 = ready_sv2 = false;
  digitalWriteFast(PIN_READY_OLF1, LOW);
  digitalWriteFast(PIN_READY_OLF2, LOW);
  digitalWriteFast(PIN_READY_SV1, LOW);
  digitalWriteFast(PIN_READY_SV2, LOW);

  enqueue_logf("MODE t_us=%lu state=SERIAL reason=CMD_STOP", micros());
  do_send_staged();
  Serial.println("OPENLOOP STOP OK");
}

static void print_help() {
  Serial.println(F(
    "Commands:\n"
    "  CTRL <OLF1|OLF2|ALL>\n"
    "  OD1..OD5 <OLF1|OLF2|ALL>\n"
    "  STOP <SV1|SV2|ALL>\n"
    "  STIM <SV1|SV2|ALL>\n"
    "  RESET\n"
    "  TEST START [on_ms [off_ms]]\n"
    "  TEST STOP\n"
    "  PRINT | STATE\n"
    "  OPENLOOP START\n"
    "  OPENLOOP STOP"
  ));
}

static void print_status() {
  Serial.print("MODE: ");
  Serial.println(currentMode == MODE_OPEN_LOOP ? "OPEN_LOOP" : "SERIAL");
  if (frameDirty) {
    rebuild_staged_frame();
  }
  Serial.print("STATE: ");
  print_state_line(stagedFrame);
  Serial.print("TEST: ");
  Serial.println(testWalker.active ? "ACTIVE" : "idle");
}

static void handle_command(char* rawLine) {
  while (*rawLine == ' ' || *rawLine == '\t') {
    ++rawLine;
  }
  if (!*rawLine) {
    return;
  }

  str_upper(rawLine);

  char* tok[MAX_TOKENS];
  int ntok = 0;
  char* token = strtok(rawLine, " \t");
  while (token && ntok < MAX_TOKENS) {
    tok[ntok++] = token;
    token = strtok(nullptr, " \t");
  }
  if (ntok == 0) {
    return;
  }

  const char* cmd = tok[0];

  if (strcmp(cmd, "HELP") == 0) {
    print_help();
    return;
  }

  if (strcmp(cmd, "STATUS") == 0) {
    print_status();
    return;
  }

  if (strcmp(cmd, "OPENLOOP") == 0) {
    if (ntok < 2) {
      Serial.println("ERR: OPENLOOP START|STOP");
      return;
    }
    if (strcmp(tok[1], "START") == 0) {
      enter_open_loop_mode();
      return;
    }
    if (strcmp(tok[1], "STOP") == 0) {
      exit_open_loop_mode();
      return;
    }
    Serial.println("ERR: OPENLOOP START|STOP");
    return;
  }

  if (currentMode == MODE_OPEN_LOOP) {
    Serial.println("ERR: open-loop mode active; use OPENLOOP STOP");
    return;
  }

  if (strcmp(cmd, "PRINT") == 0 || strcmp(cmd, "STATE") == 0) {
    if (frameDirty) {
      rebuild_staged_frame();
    }
    Serial.print("STATE: ");
    print_state_line(stagedFrame);
    return;
  }

  if (strcmp(cmd, "RESET") == 0) {
    preempt_test();
    staged = {0, 0, 0, 0};
    mark_dirty();
    do_send_staged();
    Serial.println("RESET OK");
    return;
  }

  if (strcmp(cmd, "TEST") == 0) {
    if (ntok < 2) {
      Serial.println("ERR: TEST START|STOP");
      return;
    }
    if (strcmp(tok[1], "START") == 0) {
      uint32_t onMs = (ntok >= 3 && is_numeric(tok[2])) ? (uint32_t)atol(tok[2]) : 500;
      uint32_t offMs = (ntok >= 4 && is_numeric(tok[3])) ? (uint32_t)atol(tok[3]) : 500;
      start_test(onMs, offMs);
      Serial.print("TEST START on=");
      Serial.print(onMs);
      Serial.print(" off=");
      Serial.println(offMs);
      return;
    }
    if (strcmp(tok[1], "STOP") == 0) {
      stop_test(true);
      return;
    }
    Serial.println("ERR: TEST START|STOP");
    return;
  }

  if (strcmp(cmd, "STOP") == 0 || strcmp(cmd, "STIM") == 0) {
    if (ntok < 2) {
      Serial.println("ERR: STOP/STIM needs SV1|SV2|ALL");
      return;
    }
    preempt_test();
    bool toClean = (strcmp(cmd, "STOP") == 0);
    bool sv1 = false;
    bool sv2 = false;
    if (!parse_sv_target(tok[1], sv1, sv2)) {
      Serial.println("ERR: bad switch target");
      return;
    }
    if (sv1) staged.sv1 = toClean ? 0x00 : 0x03;
    if (sv2) staged.sv2 = toClean ? 0x00 : 0x03;
    mark_dirty();
    do_send_staged();
    Serial.print(cmd);
    Serial.print(' ');
    Serial.print(tok[1]);
    Serial.println(" OK");
    return;
  }

  uint8_t firstBit;
  if (parse_olf_cmd(cmd, firstBit)) {
    if (ntok < 2) {
      Serial.println("ERR: needs OLF1|OLF2|ALL");
      return;
    }
    preempt_test();
    bool olf1 = false;
    bool olf2 = false;
    if (!parse_olf_target(tok[1], olf1, olf2)) {
      Serial.println("ERR: bad OLF target");
      return;
    }
    olf_set_pair(staged, olf1, olf2, firstBit);
    mark_dirty();
    do_send_staged();
    Serial.print(cmd);
    Serial.print(' ');
    Serial.print(tok[1]);
    Serial.println(" OK");
    return;
  }

  Serial.print("ERR: unknown command '");
  Serial.print(cmd);
  Serial.println("'");
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("=== TPIC6B595 controller v2 (Teensy 4.1) ===");
  Serial.println("Type HELP for commands. HB ON to enable heartbeat.");
  Serial.println("OPENLOOP START switches to DAQ-driven open-loop mode.");

  pinMode(PIN_CS, OUTPUT);
  SPI.begin();
}

static void process_serial_input() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      cmdBuf[cmdLen] = '\0';
      handle_command(cmdBuf);
      cmdLen = 0;
    } else if (cmdLen < MAX_CMD_BUF - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }
}

static void process_test_walker() {
  if (!testWalker.active || currentMode != MODE_SERIAL) {
    return;
  }

  if (testWalker.onPhase) {
    if (testWalker.phaseTimer >= testWalker.on_ms) {
      testWalker.onPhase = false;
      testWalker.phaseTimer = 0;
      test_send_zero();
    }
  } else if (testWalker.phaseTimer >= testWalker.off_ms) {
    testWalker.onPhase = true;
    testWalker.phaseTimer = 0;
    ++testWalker.idx;
    if (testWalker.idx >= kTestBitCount) {
      testWalker.idx = 0;
    }
    test_send_one_hot(kTestBits[testWalker.idx]);
  }
}

static void drain_logs() {
  if (logHead == logTail) {
    return;
  }

  noInterrupts();
  int idx = logTail;
  interrupts();

  Serial.println((char*)logBuffer[idx]);
  if (logFile) {
    logFile.println((char*)logBuffer[idx]);
    logFile.flush();
  }

  noInterrupts();
  logTail = (logTail + 1) % LOG_BUFFER_SIZE;
  interrupts();
}

void loop() {
  process_serial_input();
  process_test_walker();
  drain_logs();
}