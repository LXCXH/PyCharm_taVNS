// 文件名：TFLite_V1.cpp
#include "TFLite_V1.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"  // 包含 g_model[] 模型数据

namespace {
  // TFLite Micro 运行所需的 arena
  constexpr int kTensorArenaSize = 16 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];

  // 解释器及张量指针
  const tflite::Model*      model       = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor*             input       = nullptr;
  TfLiteTensor*             output      = nullptr;

  // 归一化参数（训练时提取），请保持与训练框架一致
  constexpr float glucose_min = 0.0f;
  constexpr float glucose_max = 586.0f;
}

void TFLiteV1Init() {
  // 加载模型
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema version mismatch: %d != %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // 注册算子
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddAdd();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddReshape();

  // 构建解释器
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 分配张量
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate tensors.");
    interpreter = nullptr;
    return;
  }

  // 获取输入/输出张量指针
  input  = interpreter->input(0);
  output = interpreter->output(0);
  
  // 输出初始化完成信息
  Serial.printf("✅ Glucose prediction model initialization completed\n");
  //Serial.printf("   Input shape: [%d, %d] (glucose sequence)\n", input->dims->data[0], input->dims->data[1]);
  //Serial.printf("   Output shape: [%d, %d] (predicted glucose)\n", output->dims->data[0], output->dims->data[1]);
  //Serial.printf("   Memory usage: %d / %d bytes (%.1f%%)\n", 
  //              interpreter->arena_used_bytes(), kTensorArenaSize,
  //              100.0f * interpreter->arena_used_bytes() / kTensorArenaSize);
  //Serial.printf("   Model size: %.1f KB, Ready for prediction\n", sizeof(g_model) / 1024.0f);
}

float TFLiteV1Predict(const float input_data[12]) {
  if (!interpreter) {
    Serial.println("Interpreter not initialized!");
    return NAN;
  }

  // 归一化并拷贝输入
  const float denom = (glucose_max - glucose_min);
  for (int i = 0; i < 12; ++i) {
    input->data.f[i] = (input_data[i] - glucose_min) / denom;
  }

  // 推理
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed.");
    return NAN;
  }

  // 反归一化输出
  float y_norm = output->data.f[0];
  float y_raw  = y_norm * denom + glucose_min;
  return y_raw;
}
