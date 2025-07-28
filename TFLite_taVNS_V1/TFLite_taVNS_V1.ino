#include <Arduino.h>
#include <esp_heap_caps.h>  // PSRAM 分配
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

constexpr size_t kTensorArenaSize = 300 * 1024;
uint8_t* tensor_arena = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 标准化参数（StandardScaler）
float input_mean[12] = {
  14.269312371613701, 15.379546251099335, 15.945779521946635, 16.577545470688346,
  16.654396662100815, 16.375724653371122, 16.066586048170898, 15.502436228632330,
  14.706921583073274, 14.230392666420926, 14.349488736677264, 14.393917380977754
};
float input_scale[12] = {
  9.581531504313908, 9.934215678239104, 9.748959516969576, 9.617725379601897,
  9.580969160321976, 9.270638644927608, 9.179212987596644, 8.949354072264624,
  8.458124841970958, 8.217841247475803, 8.801387915718013, 9.133663865077766
};

// MinMaxScaler 逆变换参数
// param_scaler_min 对应 sklearn.min_, param_scaler_scale 对应 sklearn.scale_ :contentReference[oaicite:3]{index=3}
float output_min[5] = {
  -0.6666666666666666, -0.6535947712418302, -1.2258064516129032,
  -0.1111111111111111, -0.36363636363636365
};
float output_scale[5] = {
  0.08333333333333333, 0.6535947712418302, 0.06451612903225806,
  0.0007936507936507937, 0.11363636363636363
};

// 测试序列
float test_sequences[3][12] = {
  {19.47, 19.82, 21.26, 22.35, 20.42, 19.73, 20.50, 18.78, 16.91, 17.62, 16.91, 17.61},
  {30.00, 30.00, 30.00, 30.00, 30.00, 30.00, 30.00, 30.00, 29.81, 27.77, 29.81, 30.00},
  {25.00, 24.73, 24.68, 25.77, 27.05, 28.68, 28.95, 27.86, 26.95, 26.14, 25.32, 24.50}
};
const int kNumSamples = 3;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F(">> Starting taVNS model inference"));

  // 在 PSRAM 分配 arena
  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
  if (!tensor_arena) {
    Serial.println(F("❌ PSRAM allocation failed!"));
    while (1);
  }

  // 加载模型
  const tflite::Model* model = tflite::GetModel(tavns_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("❌ Model schema mismatch!"));
    while (1);
  }

  // 注册算子
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddAdd();
  resolver.AddConv2D();
  resolver.AddMean();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  resolver.AddExpandDims();
  resolver.AddRelu();  // 如果模型中含有 ReLU 激活

  // 创建解释器并分配张量
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("❌ AllocateTensors failed."));
    while (1);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

  // 检查模型数据类型
  Serial.println(input->type == kTfLiteFloat32 ? "🔍 Float32 model" : "🔍 Quant model");

  // 推理每个测试样本
  for (int i = 0; i < kNumSamples; ++i) {
    Serial.printf("\n样本 %d:\n", i + 1);
    // 标准化
    float x_norm[12];
    Serial.print("  输入: [");
    for (int j = 0; j < 12; ++j) {
      float v = test_sequences[i][j];
      if (j) Serial.print(' ');
      Serial.printf("%.2f", v);
      x_norm[j] = (v - input_mean[j]) / input_scale[j];
    }
    Serial.println("]");

    // 填充输入
    if (input->type == kTfLiteFloat32) {
      for (int j = 0; j < 12; ++j) {
        input->data.f[j] = x_norm[j];
      }
    } else {  // int8 量化模型
      float q_scale = input->params.scale;
      int   q_zp    = input->params.zero_point;
      for (int j = 0; j < 12; ++j) {
        int32_t q = lround(x_norm[j] / q_scale) + q_zp;
        input->data.int8[j] = (int8_t)q;
      }
    }

    // 推理
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.printf("❌ Invoke failed at sample %d\n", i + 1);
      continue;
    }

    // 取得输出并做 MinMaxScaler 的逆变换：x_original = (x_scaled - min_) / scale_ :contentReference[oaicite:4]{index=4}
    float result[5];
    if (output->type == kTfLiteFloat32) {
      for (int k = 0; k < 5; ++k) {
        float y = output->data.f[k];
        result[k] = (y - output_min[k]) / output_scale[k];
      }
    } else {  // int8 量化模型
      float q_scale = output->params.scale;
      int   q_zp    = output->params.zero_point;
      for (int k = 0; k < 5; ++k) {
        float y = (output->data.int8[k] - q_zp) * q_scale;
        result[k] = (y - output_min[k]) / output_scale[k];
      }
    }

    // 打印最终参数
    Serial.printf(
      "  预测参数: [频率=%.2fHz, 电流=%.2fmA, 时长=%.1fmin, 脉宽=%.0fμs, 周期=%.1f周]\n",
      result[0], result[1], result[2], result[3], result[4]
    );
  }

  Serial.println("\n✅ Inference completed.");
}

void loop() {
  // 空循环
}
