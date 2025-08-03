// 文件名：TFLite_taVNS.h
#ifndef TFLITE_TAVNS_H
#define TFLITE_TAVNS_H

#include <Arduino.h>

/// taVNS参数预测结果结构体
struct taVNS_Params {
  float frequency;  // 频率 Hz
  float current;    // 电流 mA
  float duration;   // 时长 min
  float pulse_width; // 脉宽 μs
  float cycles;     // 周期
};

/// 初始化 taVNS TFLite 微型解释器，需在 setup() 中调用一次
void TFLite_taVNS_Init();

/// 对 12 个原始血糖值（mg/dL）进行taVNS参数预测
/// 注意：输入的mg/dL值会自动转换为mmol/L单位用于模型推理
/// 返回预测的taVNS参数结构体
/// 若未初始化或推理失败，返回的结构体中所有值为NAN
taVNS_Params TFLite_taVNS_Predict(const float input_data[12]);

/// 获取适合nurosymControlRoutine使用的时间和强度参数
/// time_minutes: 输出时间（分钟），会转换为整数
/// amp_ma: 输出强度（0-45区间），电流值会线性映射到此区间并转换为整数
void TFLite_taVNS_GetControlParams(const taVNS_Params& params, int* time_minutes, int* amp_ma);

#endif // TFLITE_TAVNS_H