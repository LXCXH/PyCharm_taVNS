// 文件名：TFLite_V1.h
#ifndef TFLITE_V1_H
#define TFLITE_V1_H

#include <Arduino.h>

/// 初始化 TFLite 微型解释器，需在 setup() 中调用一次
void TFLiteV1Init();

/// 对 12 个原始血糖值（mg/dL）进行预测，返回预测的原始血糖值（mg/dL）
/// 若未初始化或推理失败，返回 NAN
float TFLiteV1Predict(const float input_data[12]);

#endif // TFLITE_V1_H
