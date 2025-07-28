# ESP32-S3 taVNS参数预测模型测试

## 📋 项目概述

这是一个用于在ESP32-S3微控制器上测试taVNS（经皮耳廓迷走神经刺激）参数预测模型的简单Arduino项目。该项目使用TensorFlow Lite Micro框架，根据血糖序列数据预测最优的taVNS刺激参数，所有测试结果通过串口输出，无需额外的网络连接或复杂配置。

## 🎯 功能特性

- ✅ **简单易用**: 只需串口监视器即可查看所有测试结果
- ✅ **实时推理**: 在ESP32-S3上进行本地机器学习推理
- ✅ **多样本测试**: 包含5种不同血糖状态的测试样本
- ✅ **性能评估**: 推理时间、内存使用、准确性评估
- ✅ **基准测试**: 100次推理的性能基准测试
- ✅ **系统监控**: ESP32-S3硬件资源使用监控
- ✅ **参数验证**: 预测参数的合理性验证
- ✅ **无网络依赖**: 纯本地测试，无需WiFi或其他连接
- ✅ **内存优化**: 使用MicroMutableOpResolver仅注册必需的算子，减少内存占用
- ✅ **简化架构**: 移除冗余的错误报告器和包装头文件，直接使用TensorFlow Lite Micro核心接口

## 🔧 硬件要求

### 必需配置
- **开发板**: ESP32-S3开发板
- **Flash存储**: 至少4MB
- **RAM**: 至少512KB SRAM

### 推荐配置
- **Flash存储**: 8MB或16MB
- **PSRAM**: 8MB外部PSRAM
- **CPU频率**: 240MHz

## 📚 软件依赖

### Arduino IDE设置
1. **Arduino IDE**: 版本1.8.19或更高
2. **ESP32开发板包**: 版本2.0.0或更高

### 库依赖
- **TensorFlowLite_ESP32**: 用于TensorFlow Lite Micro推理

## 🚀 安装步骤

### 1. 安装Arduino IDE和ESP32支持

```bash
# 1. 下载并安装Arduino IDE
# 2. 在Arduino IDE中添加ESP32开发板包
# 文件 → 首选项 → 附加开发板管理器网址
# 添加: https://dl.espressif.com/dl/package_esp32_index.json

# 3. 安装ESP32开发板包
# 工具 → 开发板 → 开发板管理器 → 搜索"ESP32" → 安装
```

### 2. 安装TensorFlow Lite库

**重要**: 必须先安装TensorFlowLite_ESP32库才能编译成功

```bash
# 方法1: 通过Arduino IDE库管理器安装
# 工具 → 管理库 → 搜索"TensorFlowLite_ESP32" → 安装最新版本

# 方法2: 如果库管理器找不到，手动安装
# 1. 下载库: https://github.com/tanakamasayuki/TensorFlowLite_ESP32
# 2. 解压到Arduino libraries文件夹
# 3. 重启Arduino IDE
```

**验证安装**: 编译前确保在 工具 → 管理库 中能看到"TensorFlowLite_ESP32"库已安装

**注意**: 本项目直接使用TensorFlow Lite Micro核心头文件，不依赖包装库的头文件

### 3. 准备模型数据

#### 方法1: 使用Python脚本生成
```python
# 运行Python转换脚本
python convert_to_tflite.py

# 使用提供的Python脚本转换模型数据
def tflite_to_c_array(tflite_path, output_path):
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    with open(output_path, 'w') as f:
        f.write('#ifndef TAVNS_MODEL_DATA_H\n')
        f.write('#define TAVNS_MODEL_DATA_H\n\n')
        f.write('const unsigned char tavns_model_data[] = {\n')
        
        for i, byte in enumerate(data):
            if i % 12 == 0:
                f.write('  ')
            f.write(f'0x{byte:02x}')
            if i < len(data) - 1:
                f.write(', ')
            if (i + 1) % 12 == 0:
                f.write('\n')
        
        f.write('\n};\n\n')
        f.write(f'const unsigned int tavns_model_data_len = {len(data)};\n\n')
        f.write('#endif // TAVNS_MODEL_DATA_H\n')

# 使用方法
tflite_to_c_array('TFLite_Output/latest/tavns_model_improved.tflite', 
                  'Arduino_Test/tavns_model_data.h')
```

#### 方法2: 使用xxd命令（Linux/Mac）
```bash
# 在终端中运行
cd TFLite_Output/latest/
xxd -i tavns_model_improved.tflite > ../../Arduino_Test/tavns_model_data.h

# 编辑生成的文件，修改变量名为tavns_model_data
```

### 4. 提取标准化参数

```python
# 在Python中运行以下代码
import pickle

# 找到最新的训练输出目录
model_dir = 'Training_Outputs/training_output_YYYYMMDD_HHMMSS'  # 替换为实际路径

with open(f'{model_dir}/data_processor.pkl', 'rb') as f:
    processor = pickle.load(f)

print("// 更新scaler_params.h中的以下数值:")
print("const float glucose_scaler_mean[12] = {")
mean_values = [f"{val:.6f}" for val in processor.glucose_scaler.mean_]
print("  " + ", ".join(mean_values))
print("};")

print("\nconst float glucose_scaler_std[12] = {")
std_values = [f"{val:.6f}" for val in processor.glucose_scaler.scale_]
print("  " + ", ".join(std_values))
print("};")

print("\nconst float param_scaler_min[5] = {")
min_values = [f"{val:.6f}" for val in processor.param_scaler.data_min_]
print("  " + ", ".join(min_values))
print("};")

print("\nconst float param_scaler_max[5] = {")
max_values = [f"{val:.6f}" for val in processor.param_scaler.data_max_]
print("  " + ", ".join(max_values))
print("};")
```

## ⚙️ 配置选项

### Arduino IDE配置

```
开发板: ESP32S3 Dev Module
USB CDC On Boot: Enabled
CPU频率: 240MHz
Flash频率: 80MHz
Flash模式: QIO
Flash大小: 4MB (32Mb) 或更大
PSRAM: OPI PSRAM (如果可用)
分区方案: Default 4MB with spiffs
上传速度: 921600
```

### 性能优化选项

```cpp
// 在esp32_tavns_test.ino中可以调整的参数:

const int TENSOR_ARENA_SIZE = 60 * 1024;  // TensorFlow内存池大小
const int NUM_TEST_SAMPLES = 5;           // 测试样本数量
const int BENCHMARK_ITERATIONS = 100;    // 基准测试迭代次数

// 算子配置 (根据模型需要调整)
static tflite::MicroMutableOpResolver<8> resolver;  // 最多8个算子
// 当前注册的算子:
// - AddAdd() - 加法运算
// - AddFullyConnected() - 全连接层
// - AddLogistic() - Sigmoid激活函数
// - AddReshape() - 张量重塑
// - AddQuantize() - 量化操作
// - AddDequantize() - 反量化操作
// - AddMul() - 乘法运算
// - AddSub() - 减法运算

// 注意: 使用直接的TensorFlow Lite Micro头文件，无需包装库
// - 移除了 #include <TensorFlowLite_ESP32.h> 包装头文件
// - 移除了 MicroErrorReporter 错误报告器
// 错误信息将直接通过Serial输出
```

## 📊 预期输出

### 启动信息
```
=== ESP32-S3 taVNS参数预测模型测试 ===
初始化TensorFlow Lite Micro...
✓ 算子注册完成
✓ 模型加载成功
✓ 张量分配成功
✓ 模型初始化完成
  输入形状: [1, 12]
  输出形状: [1, 5]
  内存使用: 45120 / 61440 字节

=== 系统信息 ===
芯片型号: ESP32-S3
芯片版本: 0
CPU频率: 240 MHz
Flash大小: 8 MB
PSRAM大小: 8192 KB
可用堆内存: 298 KB
可用PSRAM: 8000 KB
```

### 测试样本输出
```
--- 测试样本 1: 正常空腹血糖 ---
输入血糖: [5.2 5.1 5.3 5.0 5.2 5.4 5.1 5.0 5.3 5.2 5.1 5.0]
预期参数: [频率=14.32Hz, 电流=1.68mA, 时长=29.3min, 脉宽=367μs, 周期=8.1周]
TFLite预测: [频率=14.32Hz, 电流=1.68mA, 时长=29.3min, 脉宽=367μs, 周期=8.1周]
原始空间MAE: 0.0000

[... 其他4个测试样本类似输出 ...]

=== 性能基准测试 ===
平均推理时间: 2145.32 μs (2.145 ms)
内存使用: 45120 / 61440 字节 (73.4%)

=== 所有测试完成 ===
测试结果已输出完成，可以查看串口监视器中的完整结果。
```

### 性能基准测试
```
=== 性能基准测试 ===
运行 100 次推理...
完成 25/100 次推理
完成 50/100 次推理
完成 75/100 次推理
完成 100/100 次推理

基准测试结果:
  平均推理时间: 2145.32 μs (2.145 ms)
  最短推理时间: 2098 μs (2.098 ms)
  最长推理时间: 2234 μs (2.234 ms)
  推理频率: 466.1 Hz

内存使用统计:
  TensorFlow Arena: 45120 / 61440 字节 (73.4%)
  系统堆内存: 156 / 327 KB (47.7%)
  PSRAM使用: 192 / 8192 KB (2.3%)
```

## 🔍 故障排除

### 常见问题

#### 1. 编译错误 - 找不到TensorFlow头文件
```
fatal error: tensorflow/lite/micro/micro_mutable_op_resolver.h: No such file or directory
```
**解决方案**: 
- 安装TensorFlowLite_ESP32库：工具 → 管理库 → 搜索"TensorFlowLite_ESP32"
- 如果找不到，手动从GitHub下载安装
- 重启Arduino IDE后再次编译

#### 2. 编译错误 - 模型数据未找到
```
错误: 'tavns_model_data' was not declared in this scope
```
**解决方案**: 确保`tavns_model_data.h`文件包含正确的模型数据

#### 3. 内存不足
```
✗ 张量分配失败
```
**解决方案**: 
- 减少`TENSOR_ARENA_SIZE`
- 启用PSRAM
- 选择更大的分区方案

#### 4. 模型加载失败
```
模型版本不匹配: 4 vs 3
```
**解决方案**: 确保使用兼容的TensorFlow Lite版本

#### 5. 推理结果异常
```
平均绝对误差: 112.5678
✗ 预测质量: 较差
```
**解决方案**: 
- 检查标准化参数是否正确
- 验证模型数据完整性
- 确认输入数据格式

#### 6. 算子不支持错误
```
Didn't find op for builtin opcode 'QUANTIZE'
Didn't find op for builtin opcode 'FULLY_CONNECTED'
```
**解决方案**: 
- 在setup()中添加缺失的算子: `resolver.AddQuantize();` `resolver.AddFullyConnected();`
- 增加MicroMutableOpResolver的容量: `<8>` 或更大数字
- 当前支持的算子: AddAdd, AddFullyConnected, AddLogistic, AddReshape, AddQuantize, AddDequantize, AddMul, AddSub

#### 7. 错误报告器编译错误
```
'MicroErrorReporter' was not declared in this scope
```
**解决方案**: 
- 新版TensorFlow Lite Micro已移除MicroErrorReporter
- 移除相关头文件: `#include "tensorflow/lite/micro/micro_error_reporter.h"`
- 移除变量声明: `tflite::MicroErrorReporter micro_error_reporter;`
- 简化解释器创建: 移除`&micro_error_reporter`参数

#### 8. 包装头文件编译错误
```
fatal error: TensorFlowLite_ESP32.h: No such file or directory
```
**解决方案**: 
- 移除包装头文件: `#include <TensorFlowLite_ESP32.h>`
- 直接使用TensorFlow Lite Micro核心头文件
- 确保已安装TensorFlowLite_ESP32库（提供核心头文件）
- 使用直接的tensorflow/lite/micro/路径引用

### 调试技巧

#### 1. 启用详细日志
```cpp
// 在setup()中添加
Serial.setDebugOutput(true);
print_scaler_info();  // 打印标准化参数
```

#### 2. 验证标准化参数
```cpp
// 在setup()中添加
if (!validate_scaler_params()) {
  Serial.println("标准化参数验证失败!");
  return;
}
```

#### 3. 内存监控
```cpp
// 定期检查内存使用
void printMemoryStatus() {
  Serial.printf("堆内存: %d KB, PSRAM: %d KB\n", 
                ESP.getFreeHeap() / 1024, ESP.getFreePsram() / 1024);
}
```

## 🔬 测试样本说明

项目包含5个预定义的测试样本，涵盖不同的血糖状态：

1. **正常空腹血糖** (5.0-5.4 mmol/L): 健康人群的空腹血糖水平
2. **糖尿病高血糖** (15-16 mmol/L): 糖尿病患者的高血糖状态
3. **餐后血糖升高** (7-12 mmol/L): 餐后血糖的正常升高
4. **血糖波动较大** (6-14 mmol/L): 血糖控制不稳定的情况
5. **低血糖状态** (3.5-3.9 mmol/L): 低血糖风险状态

每个样本都包含对应的预期taVNS参数，用于验证模型预测的准确性。

## 📈 性能指标

### 预期性能指标
- **推理时间**: < 5ms
- **内存使用**: < 80KB
- **预测精度**: MAE < 1.0（优秀），< 5.0（良好）
- **推理频率**: > 200Hz

### 实际测试结果
根据不同的ESP32-S3配置，实际性能可能有所差异。建议在您的具体硬件上运行基准测试以获得准确的性能数据。

## 🔄 自定义使用

### 添加新的测试样本
```cpp
// 在test_samples数组中添加新样本
{
  // 血糖序列 (12个值)
  {6.2, 6.1, 6.3, 6.0, 6.2, 6.4, 6.1, 6.0, 6.3, 6.2, 6.1, 6.0},
  "自定义血糖状态",
  {15.0, 1.5, 30.0, 400, 8.0}  // 预期参数
}
```

### 预测自定义血糖序列
```cpp
// 在loop()中调用
float custom_glucose[12] = {7.2, 7.1, 7.3, 7.0, 7.2, 7.4, 7.1, 7.0, 7.3, 7.2, 7.1, 7.0};
predictCustomGlucose(custom_glucose);
```

## 📝 许可证

本项目遵循开源许可证，具体许可证信息请参考主项目。

## 🤝 贡献

欢迎提交问题报告和改进建议！

## 📞 支持

如果遇到问题，请：
1. 检查本README的故障排除部分
2. 验证硬件配置和软件依赖
3. 提交详细的问题报告，包括错误信息和系统配置 