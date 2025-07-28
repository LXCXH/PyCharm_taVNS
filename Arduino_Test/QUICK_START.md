# 🚀 ESP32-S3 taVNS模型测试 - 快速开始

## ⚡ 解决编译错误的步骤

### 第1步: 安装TensorFlowLite_ESP32库

**如果出现编译错误**: `tensorflow/lite/micro/all_ops_resolver.h: No such file or directory`

1. **打开Arduino IDE**
2. **安装库**: 工具 → 管理库 → 搜索 `TensorFlowLite_ESP32`
3. **点击安装** 最新版本
4. **重启Arduino IDE**

### 第2步: 配置ESP32-S3开发板

```
开发板: ESP32S3 Dev Module
Flash大小: 32MB (256Mb)
PSRAM: OPI PSRAM
CPU频率: 240MHz
上传速度: 921600
USB CDC On Boot: Enabled
```

### 第3步: 准备必要文件

在编译前，确保以下文件存在：

```
Arduino_Test/
├── esp32_tavns_test.ino     ✓ 主程序
├── tavns_model_data.h       ⚠️ 需要生成
└── scaler_params.h          ⚠️ 需要生成
```

### 第4步: 生成模型文件

运行Python脚本生成Arduino所需文件：

```bash
cd PyCharm_taVNS
python Arduino_Test/generate_arduino_files.py
```

### 第5步: 编译上传

1. **打开** `esp32_tavns_test.ino`
2. **选择正确的开发板和端口**
3. **点击编译** (验证按钮)
4. **解决任何剩余错误**
5. **上传到ESP32-S3**

## 🔧 常见编译错误解决

### 错误1: 找不到TensorFlow库
```
fatal error: tensorflow/lite/micro/micro_mutable_op_resolver.h: No such file or directory
```
**解决**: 安装 `TensorFlowLite_ESP32` 库

### 错误1b: 包装头文件错误
```
fatal error: TensorFlowLite_ESP32.h: No such file or directory
```
**解决**: 移除 `#include <TensorFlowLite_ESP32.h>`，直接使用核心头文件

### 错误2: 找不到模型数据
```
'tavns_model_data' was not declared
```
**解决**: 运行 `generate_arduino_files.py` 生成头文件

### 错误3: 内存不足
```
region `dram0_0_seg' overflowed
```
**解决**: 选择更大的Flash分区方案 (32MB)

## 📊 成功编译后的输出

串口监视器(115200波特率)应显示：

```
=== ESP32-S3 taVNS参数预测模型测试 ===
初始化TensorFlow Lite Micro...
✓ 算子注册完成
✓ 模型加载成功
✓ 张量分配成功
✓ 模型初始化完成

--- 测试样本 1: 正常空腹血糖 ---
输入血糖: [5.2 5.1 5.3 5.0 5.2 5.4 5.1 5.0 5.3 5.2 5.1 5.0]
预期参数: [频率=14.32Hz, 电流=1.68mA, 时长=29.3min, 脉宽=367μs, 周期=8.1周]
TFLite预测: [频率=14.32Hz, 电流=1.68mA, 时长=29.3min, 脉宽=367μs, 周期=8.1周]
原始空间MAE: 0.0000

=== 性能基准测试 ===
平均推理时间: 2145.32 μs (2.145 ms)
内存使用: 45120 / 61440 字节 (73.4%)

=== 所有测试完成 ===
测试结果已输出完成，可以查看串口监视器中的完整结果。
```

## 📞 需要帮助？

1. 确认已安装 `TensorFlowLite_ESP32` 库
2. 确认已生成 `tavns_model_data.h` 和 `scaler_params.h`
3. 确认选择了正确的ESP32-S3开发板配置
4. 检查串口监视器波特率设置为115200 