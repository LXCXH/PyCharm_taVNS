#include "SPIFFSLogger.h"
#include <SPIFFS.h>
#include <Arduino.h>
#include <stdarg.h>

namespace SPIFFSLogger {

// 初始化 SPIFFS
bool begin() {
  if (!SPIFFS.begin(true)) {
    Serial.println("❌ SPIFFS Mounting failure");
    return false;
  }
  Serial.println("✅ SPIFFS Mounting successful");
  return true;
}

// 原始读数
void logRaw(uint16_t raw) {
  File f = SPIFFS.open("/log_raw.csv", FILE_APPEND);
  if (!f) {
    Serial.println("❌ Open /log_raw.csv fail");
    return;
  }
  // 正确传递两个参数：第一个是 millis(), 第二个是原始读数
  f.printf("%lu,RAW,%u\n", millis(), raw);
  f.close();
}

// 预测读数
void logPred(uint16_t pred) {
  File f = SPIFFS.open("/log_pred.csv", FILE_APPEND);
  if (!f) {
    Serial.println("❌ Open /log_pred.csv fail");
    return;
  }
  // 正确传递两个参数：millis() 和预测值
  f.printf("%lu,PRED,%u\n", millis(), pred);
  f.close();
}

// 控制决策
void logControl(bool accepted, int time_ms, int amp) {
  File f = SPIFFS.open("/log_ctrl.csv", FILE_APPEND);
  if (!f) {
    Serial.println("❌ Open /log_ctrl.csv fail");
    return;
  }
  // 依次是：millis(), "CTRL", 接受标志, 时间, 幅度
  f.printf("%lu,CTRL,%d,%d,%d\n",
           millis(),
           accepted,
           time_ms,
           amp);
  f.close();
}

// 内部：导出单个文件
static void _exportFile(const char* path) {
  Serial.printf("\n=== START %s ===\n", path);
  File f = SPIFFS.open(path, FILE_READ);
  if (!f) {
    Serial.printf("❌ Open %s fail\n", path);
    return;
  }
  while (f.available()) {
    Serial.write(f.read());
  }
  f.close();
  Serial.printf("=== END %s ===\n", path);
}

// 对外导出接口
void exportRaw()    { _exportFile("/log_raw.csv"); }
void exportPred()   { _exportFile("/log_pred.csv"); }
void exportControl(){ _exportFile("/log_ctrl.csv"); }

// 一次性导出所有日志
void exportAll() {
  exportRaw();
  exportPred();
  exportControl();
}

} // namespace SPIFFSLogger
