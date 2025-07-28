/*
 * ESP32-S3 taVNS参数预测模型测试
 * 使用TensorFlow Lite Micro进行推理
 * 
 * 硬件要求：
 * - ESP32-S3开发板
 * - 至少4MB Flash, 8MB PSRAM (推荐)
 * 
 * 库依赖：
 * - TensorFlowLite_ESP32 (通过库管理器安装)
 */

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 包含模型数据和标准化参数
#include "tavns_model_data.h"
#include "scaler_params.h"

// 模型配置常量
const int INPUT_SIZE = 12;        // 血糖序列长度
const int OUTPUT_SIZE = 5;        // 输出参数数量
const int TENSOR_ARENA_SIZE = 60 * 1024;  // 60KB内存池

// TensorFlow Lite相关变量
static tflite::MicroMutableOpResolver<8> resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 内存分配
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// 测试样本结构
struct TestSample {
  float glucose_sequence[INPUT_SIZE];
  const char* description;
  float expected_params[OUTPUT_SIZE];
};

// 测试样本数据 - 预期参数使用Float32 TFLite模型的预测结果
TestSample test_samples[] = {
  {
    // 正常空腹血糖 (5.0-5.4 mmol/L)
    {5.2, 5.1, 5.3, 5.0, 5.2, 5.4, 5.1, 5.0, 5.3, 5.2, 5.1, 5.0},
    "正常空腹血糖",
    {14.26, 1.69, 29.0, 371, 8.2}  // Float32 TFLite预测结果
  },
  {
    // 糖尿病高血糖 (15-16 mmol/L)
    {15.2, 16.1, 15.8, 16.5, 15.9, 16.2, 15.7, 16.0, 15.8, 16.1, 15.9, 16.0},
    "糖尿病高血糖", 
    {19.64, 1.94, 33.4, 367, 8.8}  // Float32 TFLite预测结果
  },
  {
    // 餐后血糖升高 (7-12 mmol/L)
    {7.0, 8.5, 10.2, 12.1, 11.8, 10.5, 9.2, 8.5, 7.8, 7.2, 6.9, 6.8},
    "餐后血糖升高",
    {14.65, 0.82, 16.0, 1182, 12.2}  // Float32 TFLite预测结果
  },
  {
    // 血糖波动较大 (6-14 mmol/L)
    {8.0, 12.5, 6.8, 14.2, 7.5, 11.8, 9.2, 13.1, 8.5, 10.8, 9.5, 11.2},
    "血糖波动较大",
    {17.36, 1.90, 31.3, 358, 8.1}  // Float32 TFLite预测结果
  },
  {
    // 低血糖状态 (3.5-3.9 mmol/L)
    {3.8, 3.5, 3.9, 3.6, 3.7, 3.8, 3.5, 3.6, 3.9, 3.7, 3.8, 3.6},
    "低血糖状态",
    {13.46, 1.63, 28.1, 353, 8.2}  // Float32 TFLite预测结果
  }
};

const int NUM_TEST_SAMPLES = sizeof(test_samples) / sizeof(TestSample);

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("=== ESP32-S3 taVNS参数预测模型测试 ===");
  Serial.println("初始化TensorFlow Lite Micro...");
  
  // 初始化TensorFlow Lite
  tflite::InitializeTarget();
  
  // 注册所需的算子
  resolver.AddAdd();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMul();
  resolver.AddSub();
  Serial.println("✓ 算子注册完成");
  
  // 验证标准化参数
  validateScalerParams();
  
  // 加载模型
  model = tflite::GetModel(tavns_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("模型版本不匹配: %d vs %d\n", 
                  model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  Serial.println("✓ 模型加载成功");
  
  // 创建解释器
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;
  
  // 分配张量
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("✗ 张量分配失败");
    return;
  }
  Serial.println("✓ 张量分配成功");
  
  // 获取输入输出张量
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // 验证张量形状
  if (input->dims->size != 2 || input->dims->data[1] != INPUT_SIZE) {
    Serial.printf("✗ 输入张量形状错误: [%d, %d], 期望: [1, %d]\n", 
                  input->dims->data[0], input->dims->data[1], INPUT_SIZE);
    return;
  }
  
  Serial.printf("✓ 模型初始化完成\n");
  Serial.printf("  输入形状: [%d, %d]\n", input->dims->data[0], input->dims->data[1]);
  Serial.printf("  输出形状: [%d, %d]\n", output->dims->data[0], output->dims->data[1]);
  Serial.printf("  输入张量类型: %d\n", input->type);
  Serial.printf("  输出张量类型: %d\n", output->type);
  Serial.printf("  内存使用: %d / %d 字节\n", 
                interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);
  
  // 显示系统信息
  printSystemInfo();
  
  // 检查模型数据完整性
  Serial.println("\n=== 模型数据检查 ===");
  Serial.printf("模型数据大小: %d 字节\n", tavns_model_data_len);
  
  // 检查模型数据的前几个字节（TFLite魔数）
  Serial.print("模型头部数据: ");
  for (int i = 0; i < 16; i++) {
    Serial.printf("0x%02X ", tavns_model_data[i]);
  }
  Serial.println();
  
  // 简单的模型测试 - 使用全零输入
  Serial.println("\n=== 模型完整性测试 ===");
  testModelIntegrity();
  
  Serial.println("\n=== 开始模型测试 ===");
}

void loop() {
  static bool tests_completed = false;
  
  if (!tests_completed) {
    // 运行所有测试样本
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
      runModelTest(i);
      delay(1000); // 每个测试间隔1秒
    }
    
    // 运行基准测试
    runBenchmarkTest();
    
    Serial.println("\n=== 所有测试完成 ===");
    Serial.println("测试结果已输出完成，可以查看串口监视器中的完整结果。");
    
    tests_completed = true;
  }
  
  // 测试完成后什么都不做，避免重复运行
  delay(10000);
}

void runModelTest(int test_index) {
  Serial.printf("\n--- 测试样本 %d: %s ---\n", 
                test_index + 1, test_samples[test_index].description);
  
  // 标准化输入数据
  float normalized_input[INPUT_SIZE];
  normalizeGlucose(test_samples[test_index].glucose_sequence, normalized_input);
  
  // 调试: 显示标准化后的输入
  Serial.print("标准化输入: [");
  for (int i = 0; i < INPUT_SIZE; i++) {
    Serial.printf("%.4f", normalized_input[i]);
    if (i < INPUT_SIZE - 1) Serial.print(" ");
  }
  Serial.println("]");
  
  unsigned long start_time = micros();
  
  // 设置输入数据
  for (int i = 0; i < INPUT_SIZE; i++) {
    input->data.f[i] = normalized_input[i];
  }
  
  // 运行推理
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("✗ 模型推理失败");
    return;
  }
  
  unsigned long inference_time = micros() - start_time;
  
  // 调试: 显示模型原始输出
  Serial.print("模型原始输出: [");
  float raw_outputs[OUTPUT_SIZE];
  
  // 根据输出张量类型读取数据
  if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      raw_outputs[i] = output->data.f[i];
      Serial.printf("%.6f", raw_outputs[i]);
      if (i < OUTPUT_SIZE - 1) Serial.print(" ");
    }
  } else if (output->type == kTfLiteInt8) {
    // 处理量化输出
    float scale = output->params.scale;
    int32_t zero_point = output->params.zero_point;
    Serial.printf("(量化参数: scale=%.6f, zero_point=%d) ", scale, zero_point);
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      // 反量化: (quantized_value - zero_point) * scale
      raw_outputs[i] = (output->data.int8[i] - zero_point) * scale;
      Serial.printf("%.6f", raw_outputs[i]);
      if (i < OUTPUT_SIZE - 1) Serial.print(" ");
    }
  } else {
    Serial.printf("未支持的张量类型: %d", output->type);
    return;
  }
  Serial.println("]");
  
  // 获取输出并反标准化
  float predicted_params[OUTPUT_SIZE];
  denormalizeParams(raw_outputs, predicted_params);
  
  // 显示结果 - 按照指定格式
  Serial.print("输入血糖: [");
  for (int i = 0; i < INPUT_SIZE; i++) {
    Serial.printf("%.1f", test_samples[test_index].glucose_sequence[i]);
    if (i < INPUT_SIZE - 1) Serial.print(" ");
  }
  Serial.println("]");
  
  Serial.printf("预期参数: [频率=%.2fHz, 电流=%.2fmA, 时长=%.1fmin, 脉宽=%.0fμs, 周期=%.1f周]\n",
                test_samples[test_index].expected_params[0],
                test_samples[test_index].expected_params[1], 
                test_samples[test_index].expected_params[2],
                test_samples[test_index].expected_params[3],
                test_samples[test_index].expected_params[4]);
  
  Serial.printf("TFLite预测: [频率=%.2fHz, 电流=%.2fmA, 时长=%.1fmin, 脉宽=%.0fμs, 周期=%.1f周]\n",
                predicted_params[0],
                predicted_params[1],
                predicted_params[2], 
                predicted_params[3],
                predicted_params[4]);
  
  // 计算误差
  float mae = calculateMAE(predicted_params, test_samples[test_index].expected_params);
  
  Serial.printf("原始空间MAE: %.4f\n", mae);
}

void runBenchmarkTest() {
  Serial.println("\n=== 性能基准测试 ===");
  
  const int num_iterations = 10; // 减少迭代次数，使输出更简洁
  unsigned long total_time = 0;
  
  // 标准化第一个测试样本
  float normalized_input[INPUT_SIZE];
  normalizeGlucose(test_samples[0].glucose_sequence, normalized_input);
  
  for (int i = 0; i < INPUT_SIZE; i++) {
    input->data.f[i] = normalized_input[i];
  }
  
  for (int i = 0; i < num_iterations; i++) {
    unsigned long start_time = micros();
    
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.printf("推理失败\n");
      continue;
    }
    
    unsigned long inference_time = micros() - start_time;
    total_time += inference_time;
  }
  
  float avg_time = total_time / (float)num_iterations;
  
  Serial.printf("平均推理时间: %.2f μs (%.3f ms)\n", avg_time, avg_time / 1000.0);
  Serial.printf("内存使用: %d / %d 字节 (%.1f%%)\n", 
                interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE,
                100.0 * interpreter->arena_used_bytes() / TENSOR_ARENA_SIZE);
}

void printSystemInfo() {
  Serial.println("\n=== 系统信息 ===");
  Serial.printf("芯片: %s, CPU: %dMHz, Flash: %dMB, PSRAM: %dKB\n", 
                ESP.getChipModel(), ESP.getCpuFreqMHz(), 
                ESP.getFlashChipSize() / (1024 * 1024), ESP.getPsramSize() / 1024);
}

void printMemoryUsage() {
  Serial.printf("可用堆内存: %d KB\n", ESP.getFreeHeap() / 1024);
  if (ESP.getPsramSize() > 0) {
    Serial.printf("可用PSRAM: %d KB\n", ESP.getFreePsram() / 1024);
  }
}

void normalizeGlucose(const float* raw_glucose, float* normalized_glucose) {
  for (int i = 0; i < INPUT_SIZE; i++) {
    normalized_glucose[i] = (raw_glucose[i] - glucose_scaler_mean[i]) / glucose_scaler_std[i];
  }
}

void denormalizeParams(const float* normalized_params, float* params) {
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    params[i] = normalized_params[i] * (param_scaler_max[i] - param_scaler_min[i]) + param_scaler_min[i];
  }
}

float calculateMAE(const float* predicted, const float* expected) {
  float mae = 0.0;
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    mae += abs(predicted[i] - expected[i]);
  }
  return mae / OUTPUT_SIZE;
}

// 用户接口函数：预测自定义血糖序列
void predictCustomGlucose(float* glucose_sequence) {
  Serial.println("\n=== 自定义血糖序列预测 ===");
  
  float normalized_input[INPUT_SIZE];
  normalizeGlucose(glucose_sequence, normalized_input);
  
  for (int i = 0; i < INPUT_SIZE; i++) {
    input->data.f[i] = normalized_input[i];
  }
  
  unsigned long start_time = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long inference_time = micros() - start_time;
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("✗ 推理失败");
    return;
  }
  
  float predicted_params[OUTPUT_SIZE];
  denormalizeParams(output->data.f, predicted_params);
  
  Serial.println("输入血糖序列:");
  Serial.print("  [");
  for (int i = 0; i < INPUT_SIZE; i++) {
    Serial.printf("%.1f", glucose_sequence[i]);
    if (i < INPUT_SIZE - 1) Serial.print(", ");
  }
  Serial.println("] mmol/L");
  
  Serial.println("推荐taVNS参数:");
  Serial.printf("  频率: %.2f Hz\n", predicted_params[0]);
  Serial.printf("  电流: %.2f mA\n", predicted_params[1]);
  Serial.printf("  时长: %.1f min\n", predicted_params[2]);
  Serial.printf("  脉宽: %.0f μs\n", predicted_params[3]);
  Serial.printf("  周期: %.1f 周\n", predicted_params[4]);
  Serial.printf("推理时间: %lu μs\n", inference_time);
} 

void validateScalerParams() {
  Serial.println("验证标准化参数...");
  
  // 检查血糖标准化参数
  Serial.print("血糖均值范围: ");
  float min_mean = glucose_scaler_mean[0], max_mean = glucose_scaler_mean[0];
  for (int i = 0; i < INPUT_SIZE; i++) {
    if (glucose_scaler_mean[i] < min_mean) min_mean = glucose_scaler_mean[i];
    if (glucose_scaler_mean[i] > max_mean) max_mean = glucose_scaler_mean[i];
  }
  Serial.printf("[%.2f, %.2f]\n", min_mean, max_mean);
  
  Serial.print("血糖标准差范围: ");
  float min_std = glucose_scaler_std[0], max_std = glucose_scaler_std[0];
  for (int i = 0; i < INPUT_SIZE; i++) {
    if (glucose_scaler_std[i] < min_std) min_std = glucose_scaler_std[i];
    if (glucose_scaler_std[i] > max_std) max_std = glucose_scaler_std[i];
  }
  Serial.printf("[%.2f, %.2f]\n", min_std, max_std);
  
  // 检查参数范围
  Serial.println("参数范围:");
  const char* param_names[] = {"频率", "电流", "时长", "脉宽", "周期"};
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    Serial.printf("  %s: [%.2f, %.2f]\n", param_names[i], 
                  param_scaler_min[i], param_scaler_max[i]);
  }
  
  Serial.println("✓ 标准化参数验证完成");
}

void testModelIntegrity() {
  Serial.println("测试1: 全零输入");
  
  // 设置全零输入
  for (int i = 0; i < INPUT_SIZE; i++) {
    input->data.f[i] = 0.0f;
  }
  
  // 运行推理
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("✗ 推理失败");
    return;
  }
  
  Serial.print("全零输入输出: [");
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    Serial.printf("%.6f", output->data.f[i]);
    if (i < OUTPUT_SIZE - 1) Serial.print(" ");
  }
  Serial.println("]");
  
  Serial.println("测试2: 全1输入");
  
  // 设置全1输入
  for (int i = 0; i < INPUT_SIZE; i++) {
    input->data.f[i] = 1.0f;
  }
  
  // 运行推理
  status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("✗ 推理失败");
    return;
  }
  
  Serial.print("全1输入输出: [");
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    Serial.printf("%.6f", output->data.f[i]);
    if (i < OUTPUT_SIZE - 1) Serial.print(" ");
  }
  Serial.println("]");
  
  Serial.println("测试3: 随机输入");
  
  // 设置随机输入
  for (int i = 0; i < INPUT_SIZE; i++) {
    input->data.f[i] = (float)(random(-1000, 1000)) / 1000.0f;
  }
  
  Serial.print("随机输入: [");
  for (int i = 0; i < INPUT_SIZE; i++) {
    Serial.printf("%.3f", input->data.f[i]);
    if (i < INPUT_SIZE - 1) Serial.print(" ");
  }
  Serial.println("]");
  
  // 运行推理
  status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("✗ 推理失败");
    return;
  }
  
  Serial.print("随机输入输出: [");
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    Serial.printf("%.6f", output->data.f[i]);
    if (i < OUTPUT_SIZE - 1) Serial.print(" ");
  }
  Serial.println("]");
  
  Serial.println("✓ 模型完整性测试完成");
} 