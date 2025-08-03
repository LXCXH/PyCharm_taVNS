// SH1106Display.h

#ifndef SH1106DISPLAY_H
#define SH1106DISPLAY_H

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Arduino.h>

class SH1106Display {
public:
  /**
   * @param address  I²C 地址（一般 0x3C 或 0x3D）
   * @param resetPin RST 引脚，若不使用则填 -1
   * @param sdaPin   I²C SDA 引脚
   * @param sclPin   I²C SCL 引脚
   */
  SH1106Display(uint8_t address = 0x3C, int8_t resetPin = -1,
                uint8_t sdaPin = 39, uint8_t sclPin = 38);

  /** 初始化显示，返回是否成功 */
  bool begin(uint32_t baudrate = 115200);

  /** 显示启动欢迎信息 */
  void showIntro(uint16_t delayMs = 500);

  /** 一次性更新所有字段 */
  void update(const char* status,
              const char* lastReading,
              const char* predict,
              const char* time,
              const char* amp);

  /** 单独更新“状态”行 */
  void updateStatus(const char* status);

  /** 单独更新“Last Reading”行 */
  void updateLastReading(const char* lastReading);

  /** 单独更新“Predict”行 */
  void updatePredict(const char* predict);

  /**
   * @brief 显示 “Time:… AMP:… Accept?” 并等待单个按键
   * @param timeStr    时间字符串
   * @param ampStr     强度字符串
   * @param pinButton  按键对应 GPIO，LOW 表示按下
   * @return true      接受（10 秒到期或未按下）
   *         false     拒绝（在10 秒内按下）
   */
  bool updateTimeAmp(const char* timeStr,
                     const char* ampStr,
                     uint8_t pinButton);

private:
  Adafruit_SH1106G display;
  uint8_t      _address;
  int8_t       _resetPin;
  uint8_t      _sdaPin, _sclPin;
};

#endif // SH1106DISPLAY_H
