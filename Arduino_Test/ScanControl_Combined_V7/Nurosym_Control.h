#ifndef NUROSYM_CONTROL_H
#define NUROSYM_CONTROL_H

#include <Arduino.h>
#include <stdint.h>

// GPIO 引脚定义（基于 ESP32-S3-DevKitC-1 开发板）
constexpr int GPIO_POWER = 7;
constexpr int GPIO_ENTER = 6;
constexpr int GPIO_UP    = 5;
constexpr int GPIO_DOWN  = 4;

// 时间持续计算的最大值（分钟）
constexpr int MAX_DURATION = 60;

// 核心控制接口：
//   time:     持续时间（分钟）
//   strength: 强度档位 (1–15)
void nurosymControlRoutine(int time, int strength);

#endif // NUROSYM_CONTROL_H
