// ScanControl.h
#ifndef SCANCONTROL_H
#define SCANCONTROL_H

#include <Arduino.h>

/// 初始化串口 + BLE，调用一次
void ScanControlInit();

/// 发起一次扫描→认证→读取，成功返回 true
bool ScanControlScanOnce();

/// 返回上次扫描后是否有新数据就绪
bool ScanControlIsDataReady();

/// 获取上次扫描得到的原始血糖值（mg/dL）
/// 仅当 ScanControlIsDataReady() 为 true 时调用
uint16_t ScanControlGetReading();

/// 清除“数据就绪”标志，准备下一次 ScanControlScanOnce()
void ScanControlClearDataReady();

#endif // SCANCONTROL_H
