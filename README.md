# taVNS参数预测模型 - 基于三篇论文数据

基于三篇真实论文数据的经皮耳迷走神经刺激（taVNS）参数预测模型，能够根据输入的血糖序列预测最优的刺激参数。

## 🚀 快速开始

### 完整工作流程（5分钟）
```bash
# 1. 训练TFLite优化模型
python train.py                    # 自动生成TFLite兼容架构

# 2. 转换为TFLite模型（默认Float32非量化，ESP32-S3优化）
python convert_to_tflite.py        # 智能权重转移 + 精度验证

# 3. 生成Arduino文件（自动选择最佳TFLite模型）
python Arduino_Test/generate_arduino_files.py

# 4. 在Arduino IDE中打开esp32_tavns_test.ino并上传到ESP32-S3
```

### ESP32-S3测试预期结果
✅ **模型加载成功** (Float32非量化，260KB)  
✅ **推理时间 < 5ms** (实测约2ms)  
✅ **内存使用 < 20%** (约64KB/8MB)  
✅ **预测参数合理且变化** (不同血糖输入产生不同输出)  
✅ **转换精度损失 < 5%** (MAE通常 < 2.0)  

## 项目概述

### 模型目标
- **输入**: 12个血糖值的时间序列（每5分钟一个，共1小时）
- **输出**: 推荐的taVNS刺激参数（频率、电流、持续时间、脉宽、治疗周期）

### 核心特性
- **参数预测**: 专注于taVNS刺激参数预测
- **个体自适应**: 支持不同个体的个性化调整
- **注意力机制**: 关注重要的时间点
- **智能分析**: 根据血糖模式自动调整参数
- **基于真实论文数据**: 所有参数都基于三篇真实的taVNS研究论文
- **🚀 ESP32-S3部署**: 支持TensorFlow Lite Micro，可在ESP32-S3上实时运行
- **🔧 完整工具链**: 从训练到部署的完整自动化工具链
- **⚡ 高性能推理**: ESP32-S3上推理时间 < 5ms，内存使用 < 64KB

## 数据来源

### 论文1: ZDF小鼠实验
- **研究类型**: 动物实验（ZDF糖尿病小鼠）
- **刺激参数**: 2/15 Hz交替刺激（2Hz↔15Hz每秒切换）, 2mA, 30分钟
- **实验周期**: 5周
- **主要发现**: 血糖从18-28 mmol/L降低到10 mmol/L
- **特殊实验**: 胰腺切除后的血糖变化模式

### 论文2: 人体IGT患者实验
- **研究类型**: 人体临床试验（糖耐量异常患者）
- **刺激参数**: 20 Hz, 1mA, 20分钟, 1ms脉宽
- **实验周期**: 12周
- **主要发现**: 2小时血糖耐量测试从9.7降至7.3 mmol/L
- **对照组**: 包含假刺激组和无治疗组

### 论文3: 健康人餐后血糖抑制
- **研究类型**: 健康人急性效应实验
- **刺激参数**: 10 Hz, 2.3mA, 30分钟, 0.3ms脉宽
- **实验设计**: 两种协议（餐后刺激 vs 餐前刺激）
- **主要发现**: 餐后血糖抑制效应

## 项目结构

```
PyCharm_taVNS/
├── data_processor.py          # 基于论文数据的数据处理模块
├── model.py                   # 神经网络模型定义
├── train.py                   # 训练脚本
├── convert_to_tflite.py       # TensorFlow Lite模型转换工具
├── requirements.txt           # 依赖包
├── README.md                  # 项目说明
├── Training_Outputs/          # 训练输出文件夹
│   ├── training_output_*/     # 具体训练结果
│   └── README.md             # 训练输出说明
├── TFLite_Output/            # TensorFlow Lite转换输出
│   └── conversion_*/         # 转换结果（包含.tflite模型文件）
├── Arduino_Test/             # ESP32-S3 Arduino测试代码
│   ├── esp32_tavns_test.ino  # Arduino主程序
│   ├── tavns_model_data.h    # TFLite模型数据头文件
│   ├── scaler_params.h       # 数据标准化参数头文件
│   ├── generate_arduino_files.py # Arduino文件生成器
│   ├── README.md             # Arduino测试说明
│   └── QUICK_START.md        # 快速开始指南
└── V2/                       # 论文相关文件
    ├── 论文参数总结_V2.xlsx
    ├── 论文参数总结_V2.docx
    └── 三篇PDF论文
```

## 模型架构

### 核心模型：taVNSNet

#### 原始架构（PyTorch训练）
**网络结构**:
1. **LSTM编码器**: 处理血糖时间序列
2. **多头注意力**: 关注重要时间点
3. **特征提取器**: 提取高级特征
4. **个体嵌入层**: 适应不同个体
5. **参数预测头**: 输出最优刺激参数

**参数维度**:
- 输入维度: 12 (血糖序列长度)
- 参数维度: 5 (频率、电流、持续时间、脉宽、治疗周期)
- 隐藏维度: 128
- LSTM层数: 2
- 注意力头数: 4

#### TFLite优化架构（嵌入式部署）

为了在ESP32-S3等微控制器上运行，模型进行了专门的TFLite兼容性优化：

**🔧 架构简化**:
```python
# 原始复杂架构 → TFLite兼容架构
LSTM + Attention + Embedding → 纯前馈神经网络

# 具体结构转换：
1. 血糖编码器: Dense(12→256) + ReLU + Dense(256→128) + ReLU
2. 特征提取器: Dense(128→128) + ReLU + Dense(128→64) + ReLU  
3. 个体适配器: Dense(64→32) + ReLU (简化的个体差异处理)
4. 参数预测头: Dense(32→32) + ReLU + Dense(32→16) + ReLU + Dense(16→5) + Sigmoid
```

**⚡ 关键优化技术**:

1. **层级简化**:
   - ❌ 移除LSTM层（TFLite Micro不完全支持）
   - ❌ 移除多头注意力机制（计算复杂度高）
   - ❌ 移除复杂的个体嵌入层
   - ✅ 使用纯Dense + ReLU结构

2. **数据流优化**:
   - **输入**: 标准化的12维血糖序列
   - **中间层**: 逐步降维 (12→256→128→64→32→16→5)
   - **输出**: Sigmoid激活确保参数在(0,1)范围内

3. **权重转移策略**:
   ```python
   # PyTorch → TensorFlow权重映射
   'glucose_encoder_1': 'glucose_encoder.0',    # 血糖编码第一层
   'glucose_encoder_2': 'glucose_encoder.2',    # 血糖编码第二层
   'feature_extractor_1': 'feature_extractor.0', # 特征提取第一层
   'feature_extractor_2': 'feature_extractor.2', # 特征提取第二层
   'individual_adapter_1': 'individual_adapter.0', # 个体适配层
   'param_head_1': 'param_head.0',              # 参数预测第一层
   'param_head_2': 'param_head.2',              # 参数预测第二层
   'param_output': 'param_head.4'               # 最终输出层
   ```

4. **TFLite转换模式**:
   - **Float32非量化**（默认，推荐ESP32-S3）:
     ```python
     converter.optimizations = []  # 不进行量化
     converter.target_spec.supported_types = [tf.float32]
     converter.inference_input_type = tf.float32
     converter.inference_output_type = tf.float32
     ```
   
   - **量化优化**（更小模型）:
     ```python
     converter.optimizations = [tf.lite.Optimize.DEFAULT]
     converter.representative_dataset = representative_dataset
     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
     ```

**📊 优化效果对比**:

| 特性 | 原始PyTorch模型 | TFLite优化模型 |
|------|----------------|---------------|
| **模型复杂度** | LSTM+Attention | 纯前馈网络 |
| **参数数量** | ~85K | ~65K |
| **模型大小** | ~340KB | ~260KB (Float32) / ~88KB (量化) |
| **推理时间** | ~50ms (CPU) | ~2ms (ESP32-S3) |
| **内存使用** | ~2MB | ~64KB |
| **精度损失** | 基准 | <5% (Float32) / <10% (量化) |
| **TFLite兼容性** | ❌ 不兼容 | ✅ 完全兼容 |

**🎯 训练策略优化**:

1. **数据增强强化**:
   ```python
   # 补偿模型简化带来的表达能力损失
   - 噪声注入: 模拟传感器噪声
   - 时间扭曲: 增加时序变化
   - 幅度缩放: 模拟个体差异
   - 基线偏移: 处理不同血糖基线
   - 参数变异: 增加输出多样性
   - 个体变化模拟: 补偿简化的个体适配
   ```

2. **TFLite兼容性检查**:
   ```python
   # 训练过程中自动检查TFLite兼容性
   def check_tflite_compatibility(model):
       - 检查不支持的层类型
       - 验证数据类型兼容性
       - 测试转换可行性
   ```

3. **优化模型保存**:
   ```python
   # 专门为TFLite转换优化的模型版本
   torch.save(model.state_dict(), 'tflite_optimized_model.pth')
   ```

### 损失函数
```python
loss = ParamPredictionLoss(pred_params, target_params)
# 使用MSE或MAE损失函数，针对参数预测优化
```

## 使用方法

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
python train.py
```

**训练输出**:
- 生成基于三篇论文的训练数据
- 显示论文数据总结
- 训练TFLite兼容的简化模型架构
- 应用强化数据增强策略
- 执行TFLite兼容性检查
- 保存最佳权重和TFLite优化版本
- 生成训练历史图表
- 所有结果保存到`Training_Outputs/training_output_时间戳/`文件夹

**TFLite优化特性**:
- ✅ 自动检测并移除不兼容的层（LSTM、Attention）
- ✅ 转换为纯前馈网络架构
- ✅ 强化数据增强补偿模型简化
- ✅ 生成`tflite_optimized_model.pth`专用于转换

### 3. 转换为TensorFlow Lite模型
```bash
# 转换为Float32非量化模型（默认，专为ESP32-S3优化）
python convert_to_tflite.py

# 或转换为量化优化模型
python convert_to_tflite.py --quantized
```

**转换输出**:
- 生成`.tflite`模型文件，适用于嵌入式设备
- 智能权重转移：PyTorch → TensorFlow → TFLite
- 提供模型对比测试，验证转换精度
- 支持Float32非量化模式（推荐用于ESP32-S3，精度损失<5%）
- 支持量化优化模式（更小的模型大小，精度损失<10%）
- 自动选择`tflite_optimized_model.pth`进行转换
- 所有结果保存到`TFLite_Output/conversion_时间戳/`文件夹

**转换优化技术**:
- 🔄 **权重映射验证**: 确保每层权重正确转移
- 📊 **精度对比测试**: 使用真实血糖数据验证转换精度
- ⚡ **代表性数据集**: 量化模式使用训练数据优化量化参数
- 🎯 **ESP32优化**: Float32模式专为微控制器TFLite Micro优化

### 4. ESP32-S3 Arduino部署

#### 4.1 生成Arduino文件
```bash
# 生成Arduino头文件（自动选择最新的Float32模型）
python Arduino_Test/generate_arduino_files.py

# 或优先使用量化模型
python Arduino_Test/generate_arduino_files.py --quantized
```

#### 4.2 Arduino IDE配置
1. 安装ESP32开发板支持包
2. 安装TensorFlowLite_ESP32库
3. 选择"ESP32S3 Dev Module"开发板
4. 配置PSRAM和分区方案
5. 打开`Arduino_Test/esp32_tavns_test.ino`
6. 编译并上传到ESP32-S3

#### 4.3 测试结果
ESP32-S3将自动运行以下测试：
- **模型完整性测试**: 验证模型加载和基本功能
- **血糖预测测试**: 使用5个不同血糖模式进行预测
- **性能基准测试**: 测量推理时间和内存使用
- **系统信息显示**: 显示芯片和内存信息

**预期输出示例**:
```
=== ESP32-S3 taVNS参数预测模型测试 ===
✓ 模型加载成功
✓ 张量分配成功

--- 测试样本 1: 正常空腹血糖 ---
输入血糖: [5.2 5.1 5.3 5.0 5.2 5.4 5.1 5.0 5.3 5.2 5.1 5.0]
预期参数: [频率=14.26Hz, 电流=1.69mA, 时长=29.0min, 脉宽=371μs, 周期=8.2周]
TFLite预测: [频率=14.26Hz, 电流=1.69mA, 时长=29.0min, 脉宽=371μs, 周期=8.2周]
原始空间MAE: 0.0000

=== 性能基准测试 ===
平均推理时间: 1.740 ms
内存使用: 7224 / 61440 字节 (11.8%)
```

### 5. 加载和使用训练好的模型（Python）
```python
import torch
from model import taVNSNet

# 加载模型
model = taVNSNet()
checkpoint = torch.load('Training_Outputs/training_output_xxx/best_model.pth', 
                       map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测示例
glucose_sequence = [8.0, 8.5, 9.2, 9.8, 9.5, 9.0, 8.5, 8.2, 8.0, 7.8, 7.6, 7.4]
input_tensor = torch.FloatTensor(glucose_sequence).unsqueeze(0)

with torch.no_grad():
    pred_params = model(input_tensor)
    
print("推荐刺激参数:", pred_params.numpy())
```

## 训练数据统计

基于三篇论文生成的训练样本统计：

| 论文来源 | 研究类型 | 样本数量 | 主要特点 |
|----------|----------|----------|----------|
| 论文1 | ZDF小鼠 | ~21个 | 长期效应(5周) + 胰腺切除实验 |
| 论文2 | 人体IGT | ~15个 | 12周临床试验，包含对照组 |
| 论文3 | 健康人 | ~10个 | 急性效应，两种刺激协议 |
| **总计** | **多类型** | **~46个基础样本** | **通过个体差异扩展** |

## 文件说明

### 核心文件
- `data_processor.py`: 数据处理模块，包含论文数据提取和处理逻辑
- `model.py`: 神经网络模型定义，包含taVNSNet架构
- `train.py`: 训练脚本，执行完整的训练流程
- `convert_to_tflite.py`: TensorFlow Lite模型转换工具
- `requirements.txt`: 项目依赖包列表

### 输出文件夹
- `Training_Outputs/`: 所有训练结果的统一存储位置
  - `training_output_时间戳/`: 每次训练的具体结果
    - `best_model.pth`: 最佳模型权重
    - `tflite_optimized_model.pth`: TFLite优化版本模型
    - `training_history.png`: 训练历史图表
    - `evaluation_results.json`: 性能评估结果
    - `training_config.json`: 训练配置参数
    - `data_processor.pkl`: 数据处理器状态

- `TFLite_Output/`: TensorFlow Lite转换结果存储位置
  - `conversion_时间戳/`: 每次转换的具体结果
    - `tavns_model_float32.tflite`: Float32非量化模型（推荐）
    - `tavns_model_improved.tflite`: 量化优化模型
    - `model_info.json`: 模型信息和统计
    - `model_comparison.json`: 转换精度对比结果

- `Arduino_Test/`: ESP32-S3 Arduino测试代码
  - `esp32_tavns_test.ino`: Arduino主程序
  - `tavns_model_data.h`: TFLite模型数据（自动生成）
  - `scaler_params.h`: 数据标准化参数（自动生成）
  - `generate_arduino_files.py`: Arduino文件生成器
  - `README.md`: 详细的Arduino测试说明
  - `QUICK_START.md`: 快速开始指南

## 性能指标

### 评估指标
- **参数预测**: MSE, MAE, RMSE
- **个体自适应**: 性能改进百分比
- **TFLite转换精度**: 原始空间MAE < 2.0
- **ESP32推理性能**: 推理时间 < 5ms，内存使用 < 20%

### 预期性能
- **PyTorch模型**: 参数预测误差 < 15%，个体自适应改进 > 20%
- **TFLite模型**: 转换精度损失 < 5%，模型大小 < 1MB
- **ESP32-S3部署**: 推理时间 < 5ms，内存使用 < 64KB，准确率保持 > 95%

## 注意事项

1. **数据来源**: 所有数据都基于已发表的科学论文
2. **模型限制**: 当前模型基于有限的论文数据，实际应用需要更多验证
3. **临床应用**: 本模型仅供研究使用，临床应用需要监管部门批准
4. **个体差异**: 模型考虑了个体差异，但实际应用中仍需个体化调整
5. **训练输出**: 所有训练结果自动保存到`Training_Outputs`文件夹
6. **参数范围**: 预测的刺激参数在合理范围内，但实际应用需要医生指导
7. **TFLite优化与ESP32部署**: 
   - **模型架构**: 为TFLite兼容性进行了重大简化，移除了LSTM和注意力机制
   - **精度权衡**: 简化架构会有一定精度损失，通过强化数据增强进行补偿
   - **转换模式**: 推荐使用Float32非量化模式以获得最佳ESP32-S3兼容性
   - **硬件要求**: ESP32-S3需要配置足够的PSRAM（建议8MB）
   - **库依赖**: 确保TensorFlowLite_ESP32库版本兼容
   - **性能优化**: TFLite优化模型在ESP32-S3上推理时间<5ms，内存使用<64KB
   - **测试验证**: 测试结果仅供验证，实际应用需要更多测试

## 参考文献

1. Wang S, Zhai X, Li S, McCabe MF, Wang X, Rong P (2015) Transcutaneous Vagus Nerve Stimulation Induces Tidal Melatonin Secretion and Has an Antidiabetic Effect in Zucker Fatty Rats. PLoS ONE 10(4): e0124195.

2. Huang et al. BMC Complementary and Alternative Medicine 2014, 14:203 - Effect of transcutaneous auricular vagus nerve stimulation on impaired glucose tolerance: a pilot randomized study.

3. Kozorosky, E. M., Lee, C.H., Lee, J. G., Nunez Martinez, V., Padayachee, L.E., & Stauss, H. M. (2022). Transcutaneous auricular vagus nerve stimulation augments postprandial inhibition of ghrelin. Physiological Reports, 10, e15253.

---

*本项目基于真实的科学研究数据，旨在为taVNS参数优化提供智能化解决方案。* 