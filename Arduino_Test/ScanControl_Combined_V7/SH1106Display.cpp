#include "esp32-hal-gpio.h"
// SH1106Display.cpp
#include "SH1106Display.h"

SH1106Display::SH1106Display(uint8_t address, int8_t resetPin,
                             uint8_t sdaPin, uint8_t sclPin)
  : display(128, 64, &Wire, resetPin)
  , _address(address)
  , _resetPin(resetPin)
  , _sdaPin(sdaPin)
  , _sclPin(sclPin)
{}

bool SH1106Display::begin(uint32_t baudrate) {
  Serial.begin(baudrate);
  delay(500);
  Wire.begin(_sdaPin, _sclPin);
  if (!display.begin(_address, true)) {
    Serial.println(F("OLED initialization failure"));
    return false;
  }
  Serial.println(F("✅ OLED Initialization successful"));
  display.clearDisplay();
  return true;
}

void SH1106Display::showIntro(uint16_t delayMs) {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SH110X_WHITE);
  display.setCursor(0, 0);
  display.println(F("taVNS Control"));
  display.setCursor(20, 0);
  display.println(F("Strating..."));
  display.display();
  delay(delayMs);
}

void SH1106Display::update(const char* status,
                           const char* lastReading,
                           const char* predict,
                           const char* time,
                           const char* amp) {
  display.clearDisplay();
  // 状态
  display.setTextSize(2);
  display.setCursor(0, 0);
  display.println(status);
  // Last Reading
  display.setTextSize(1);
  display.setCursor(0, 20);
  display.print(F("Reading:"));
  display.print(lastReading);
  display.println(F(" mg/dL"));
  // Predict
  display.setCursor(0, 32);
  display.print(F("Predict:"));
  display.print(predict);
  display.println(F(" mg/dL"));
  // Time & AMP
  display.setCursor(0, 44);
  display.print(F("Time:"));
  
  // 时间数值反色显示（基于中心点计算）
  int timeX = 30; // "Time:" 大约30像素宽
  
  // 获取文字实际边界
  int16_t x1, y1;
  uint16_t textWidth, textHeight;
  display.getTextBounds(time, 0, 0, &x1, &y1, &textWidth, &textHeight);
  
  // 计算反色框位置（中心对齐，左右各加一点边距）
  int rectX = timeX - 1;
  int rectWidth = textWidth + 2;
  display.fillRect(rectX, 43, rectWidth, 9, SH110X_WHITE);
  
  display.setCursor(timeX, 44);
  display.setTextColor(SH110X_BLACK);
  display.print(time);
  display.setTextColor(SH110X_WHITE);
  
  // AMP标签（基于时间数值实际宽度）
  int ampLabelX = timeX + textWidth + 5;
  display.setCursor(ampLabelX, 44);
  display.setTextColor(SH110X_WHITE);
  display.print(F("  AMP:"));
  
  // AMP数值反色显示（基于中心点计算）
  int ampX = ampLabelX + 36; // "  AMP:" 大约36像素宽
  
  // 获取AMP数值文字实际边界
  int16_t ax1, ay1;
  uint16_t ampTextWidth, ampTextHeight;
  display.getTextBounds(amp, 0, 0, &ax1, &ay1, &ampTextWidth, &ampTextHeight);
  
  // 计算反色框位置
  int ampRectX = ampX - 1;
  int ampRectWidth = ampTextWidth + 2;
  display.fillRect(ampRectX, 43, ampRectWidth, 9, SH110X_WHITE);
  
  display.setCursor(ampX, 44);
  display.setTextColor(SH110X_BLACK);
  display.print(amp);
  display.setTextColor(SH110X_WHITE);
  // footer
  display.setCursor(0, 56);
  display.println(F("Accept?"));
  display.display();
}

void SH1106Display::updateStatus(const char* status) {
  // 清除上次状态行区域（0,0）到（128,16）
  display.fillRect(0, 0, 128, 16, SH110X_BLACK);
  display.setTextSize(2);
  display.setCursor(0, 0);
  display.setTextColor(SH110X_WHITE);
  display.println(status);
  display.display();
}

void SH1106Display::updateLastReading(const char* lastReading) {
  display.fillRect(0, 20, 128, 12, SH110X_BLACK);
  display.setTextSize(1);
  display.setCursor(0, 20);
  display.setTextColor(SH110X_WHITE);
  display.print(F("Reading:"));
  display.print(lastReading);
  display.println(F(" mg/dL"));
  display.display();
}

void SH1106Display::updatePredict(const char* predict) {
  display.fillRect(0, 32, 128, 12, SH110X_BLACK);
  display.setTextSize(1);
  display.setCursor(0, 32);
  display.setTextColor(SH110X_WHITE);
  display.print(F("Predict:"));
  display.print(predict);
  display.println(F(" mg/dL"));
  display.display();
}

bool SH1106Display::updateTimeAmp(const char* timeStr,
                                  const char* ampStr,
                                  uint8_t pinButton) {
  // 绘制时间与强度
  display.fillRect(0, 44, 128, 20, SH110X_BLACK);
  display.setTextSize(1);
  display.setCursor(0, 44);
  display.setTextColor(SH110X_WHITE);
  display.print(F("Time:"));
  
  // 计算时间数值位置并绘制反色背景（基于中心点）
  int timeX = 30; // "Time:" 大约30像素宽
  
  // 获取时间数值文字实际边界
  int16_t tx1, ty1;
  uint16_t timeTextWidth, timeTextHeight;
  display.getTextBounds(timeStr, 0, 0, &tx1, &ty1, &timeTextWidth, &timeTextHeight);
  
  // 计算反色框位置
  int timeRectX = timeX - 1;
  int timeRectWidth = timeTextWidth + 2;
  display.fillRect(timeRectX, 43, timeRectWidth, 9, SH110X_WHITE);
  
  display.setCursor(timeX, 44);
  display.setTextColor(SH110X_BLACK);
  display.print(timeStr);
  display.setTextColor(SH110X_WHITE);
  
  // 绘制AMP标签（基于时间数值实际宽度）
  int ampLabelX = timeX + timeTextWidth + 10;
  display.setCursor(ampLabelX, 44);
  display.setTextColor(SH110X_WHITE);
  display.print(F("  AMP:"));
  
  // 计算AMP数值位置并绘制反色背景（基于中心点）
  int ampX = ampLabelX + 36; // "  AMP:" 大约36像素宽
  
  // 获取AMP数值文字实际边界
  int16_t ax1, ay1;
  uint16_t ampStrTextWidth, ampStrTextHeight;
  display.getTextBounds(ampStr, 0, 0, &ax1, &ay1, &ampStrTextWidth, &ampStrTextHeight);
  
  // 计算反色框位置
  int ampStrRectX = ampX - 1;
  int ampStrRectWidth = ampStrTextWidth + 2;
  display.fillRect(ampStrRectX, 43, ampStrRectWidth, 9, SH110X_WHITE);
  
  display.setCursor(ampX, 44);
  display.setTextColor(SH110X_BLACK);
  display.print(ampStr);
  display.setTextColor(SH110X_WHITE);
  display.setCursor(0, 57);
  display.println(F("Press to Reject (10s)"));
  display.display();

  // 按键配置
  pinMode(pinButton, INPUT_PULLUP);

  // 等待按键或超时
  unsigned long start = millis();
  while (millis() - start < 10000UL) {
    if (digitalRead(pinButton) == LOW) {
      // 等待释放
      while (digitalRead(pinButton) == LOW) { delay(10); }
      return false;  // 用户拒绝
    }
    delay(20);
  }
  return true;  // 超时自动接受
}
