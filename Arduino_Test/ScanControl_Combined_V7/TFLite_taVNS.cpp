// 文件名：TFLite_taVNS.cpp
#include "TFLite_taVNS.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tavns_model_data.h"  // 包含 tavns_model_data[] 模型数据
#include "scaler_params.h"     // 包含标准化参数

namespace {
  // TFLite Micro 运行所需的 arena (taVNS模型需要更大的内存)
  constexpr int kTensorArenaSize = 60 * 1024;  // 60KB
  static uint8_t tensor_arena[kTensorArenaSize];

  // 解释器及张量指针
  const tflite::Model*      model       = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor*             input       = nullptr;
  TfLiteTensor*             output      = nullptr;

  // 模型配置常量
  constexpr int INPUT_SIZE = 12;
  constexpr int OUTPUT_SIZE = 5;
}

void TFLite_taVNS_Init() {
  // Serial.println("Initializing taVNS parameter prediction model...");
  
  // 加载模型
  model = tflite::GetModel(tavns_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("taVNS model version mismatch: %d != %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // 注册算子 (taVNS模型需要更多算子)
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddAdd();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMul();
  resolver.AddSub();

  // 构建解释器
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 分配张量
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("taVNS model tensor allocation failed");
    interpreter = nullptr;
    return;
  }

  // 获取输入/输出张量指针
  input  = interpreter->input(0);
  output = interpreter->output(0);

  // 验证张量形状
  if (input->dims->size != 2 || input->dims->data[1] != INPUT_SIZE) {
    Serial.printf("taVNS input tensor shape error: [%d, %d], expected: [1, %d]\n", 
                  input->dims->data[0], input->dims->data[1], INPUT_SIZE);
    interpreter = nullptr;
    return;
  }

  Serial.printf("✅ taVNS model initialization completed\n");
  //Serial.printf("  Input shape: [%d, %d]\n", input->dims->data[0], input->dims->data[1]);
  //Serial.printf("  Output shape: [%d, %d]\n", output->dims->data[0], output->dims->data[1]);
  //Serial.printf("  Memory usage: %d / %d bytes\n", 
  //              interpreter->arena_used_bytes(), kTensorArenaSize);
}

taVNS_Params TFLite_taVNS_Predict(const float input_data[12]) {
  taVNS_Params result;
  
  // 初始化为NAN，表示错误状态
  result.frequency = NAN;
  result.current = NAN;
  result.duration = NAN;
  result.pulse_width = NAN;
  result.cycles = NAN;
  
  if (!interpreter) {
    Serial.println("taVNS interpreter not initialized!");
    return result;
  }

  // 单位转换：mg/dL -> mmol/L (除以18.0182)
  // taVNS模型训练时使用的是mmol/L单位
  const float MG_DL_TO_MMOL_L = 1.0f / 18.0182f;
  float glucose_mmol[INPUT_SIZE];
  
  Serial.print("Glucose unit conversion mg/dL -> mmol/L: [");
  for (int i = 0; i < INPUT_SIZE; i++) {
    glucose_mmol[i] = input_data[i] * MG_DL_TO_MMOL_L;
    Serial.printf("%.1f->%.2f", input_data[i], glucose_mmol[i]);
    if (i < INPUT_SIZE - 1) Serial.print(" ");
  }
  Serial.println("]");

  // 标准化输入数据并拷贝到输入张量
  for (int i = 0; i < INPUT_SIZE; i++) {
    float normalized_value;
    normalize_glucose_value(glucose_mmol[i], i, &normalized_value);
    input->data.f[i] = normalized_value;
  }

  // 推理
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("taVNS inference failed");
    return result;
  }

  // 获取输出并反标准化
  float raw_outputs[OUTPUT_SIZE];
  
  // 根据输出张量类型读取数据
  if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      raw_outputs[i] = output->data.f[i];
    }
  } else if (output->type == kTfLiteInt8) {
    // 处理量化输出
    float scale = output->params.scale;
    int32_t zero_point = output->params.zero_point;
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      // 反量化: (quantized_value - zero_point) * scale
      raw_outputs[i] = (output->data.int8[i] - zero_point) * scale;
    }
  } else {
    Serial.printf("taVNS unsupported output tensor type: %d\n", output->type);
    return result;
  }

  // 反标准化参数
  float predicted_params[OUTPUT_SIZE];
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    denormalize_param_value(raw_outputs[i], i, &predicted_params[i]);
  }

  // 填充结果结构体
  result.frequency = predicted_params[0];    // 频率 Hz
  result.current = predicted_params[1];      // 电流 mA
  result.duration = predicted_params[2];     // 时长 min
  result.pulse_width = predicted_params[3];  // 脉宽 μs
  result.cycles = predicted_params[4];       // 周期

  return result;
}

void TFLite_taVNS_GetControlParams(const taVNS_Params& params, int* time_minutes, int* amp_ma) {
  // 检查输入参数是否有效
  if (isnan(params.duration) || isnan(params.current)) {
    Serial.println("taVNS parameters invalid, using default values");
    *time_minutes = 10;  // 默认时间
    *amp_ma = 10;        // 默认电流
    return;
  }

  // 将时长转换为整数分钟 (四舍五入)
  *time_minutes = (int)(params.duration + 0.5f);
  
  // 将电流参数映射到0-45区间
  // 模型电流范围: 0.478536 - 4.370435 mA (从scaler_params.h)
  // 目标范围: 0 - 45 (整数)
  const float current_min = 0.478536f;
  const float current_max = 4.370435f;
  const int target_min = 0;
  const int target_max = 45;
  
  // 限制电流在有效范围内
  float current_clamped = params.current;
  if (current_clamped < current_min) current_clamped = current_min;
  if (current_clamped > current_max) current_clamped = current_max;
  
  // 线性映射: y = (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min
  float mapped_amp = (current_clamped - current_min) * (target_max - target_min) / (current_max - current_min) + target_min;
  *amp_ma = (int)(mapped_amp + 0.5f);  // 四舍五入

  // 安全范围检查和限制
  if (*time_minutes < 1) *time_minutes = 1;
  if (*time_minutes > 60) *time_minutes = 60;  // 最长60分钟
  
  if (*amp_ma < target_min) *amp_ma = target_min;
  if (*amp_ma > target_max) *amp_ma = target_max;

  Serial.printf("taVNS predicted parameters: freq=%.2fHz, current=%.2fmA, duration=%.1fmin, pulse_width=%.0fμs, cycles=%.1f\n",
                params.frequency, params.current, params.duration, params.pulse_width, params.cycles);
  Serial.printf("Current mapping: %.2fmA -> %d (0-45 range)\n", params.current, *amp_ma);
  Serial.printf("Final control parameters: time=%d minutes, intensity=%d (0-45 range)\n", *time_minutes, *amp_ma);
}