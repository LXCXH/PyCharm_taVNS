#ifndef SPIFFSLOGGER_H
#define SPIFFSLOGGER_H

#include <Arduino.h>

namespace SPIFFSLogger {
  // 在 setup() 里调用，挂载 SPIFFS
  bool begin();

  // 三种日志接口
  void logRaw(uint16_t raw);
  void logPred(uint16_t pred);
  void logControl(bool accepted, int time_ms, int amp);

  // 串口导出接口
  void exportRaw();
  void exportPred();
  void exportControl();
  void exportAll();
}

#endif // SPIFFSLOGGER_H
