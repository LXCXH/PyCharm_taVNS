# taVNS参数预测模型 - 基于三篇论文数据

基于三篇真实论文数据的经皮耳迷走神经刺激（taVNS）参数预测模型，能够根据输入的血糖序列预测最优的刺激参数。

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
├── requirements.txt           # 依赖包
├── README.md                  # 项目说明
├── Training_Outputs/          # 训练输出文件夹
│   ├── training_output_*/     # 具体训练结果
│   └── README.md             # 训练输出说明
└── V2/                       # 论文相关文件
    ├── 论文参数总结_V2.xlsx
    ├── 论文参数总结_V2.docx
    └── 三篇PDF论文
```

## 模型架构

### 核心模型：taVNSNet

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

### 损失函数
```python
loss = ParamPredictionLoss(pred_params, target_params)
# 使用MSE或MAE损失函数
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
- 训练模型并保存最佳权重
- 生成训练历史图表
- 所有结果保存到`Training_Outputs/training_output_时间戳/`文件夹

### 3. 加载和使用训练好的模型
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
- `requirements.txt`: 项目依赖包列表

### 输出文件夹
- `Training_Outputs/`: 所有训练结果的统一存储位置
- `training_output_时间戳/`: 每次训练的具体结果
  - `best_model.pth`: 最佳模型权重
  - `training_history.png`: 训练历史图表
  - `evaluation_results.json`: 性能评估结果
  - `training_config.json`: 训练配置参数
  - `data_processor.pkl`: 数据处理器状态

## 性能指标

### 评估指标
- **参数预测**: MSE, MAE, RMSE
- **个体自适应**: 性能改进百分比

### 预期性能
- 参数预测误差: < 15%
- 个体自适应改进: > 20%

## 注意事项

1. **数据来源**: 所有数据都基于已发表的科学论文
2. **模型限制**: 当前模型基于有限的论文数据，实际应用需要更多验证
3. **临床应用**: 本模型仅供研究使用，临床应用需要监管部门批准
4. **个体差异**: 模型考虑了个体差异，但实际应用中仍需个体化调整
5. **训练输出**: 所有训练结果自动保存到`Training_Outputs`文件夹
6. **参数范围**: 预测的刺激参数在合理范围内，但实际应用需要医生指导

## 参考文献

1. Wang S, Zhai X, Li S, McCabe MF, Wang X, Rong P (2015) Transcutaneous Vagus Nerve Stimulation Induces Tidal Melatonin Secretion and Has an Antidiabetic Effect in Zucker Fatty Rats. PLoS ONE 10(4): e0124195.

2. Huang et al. BMC Complementary and Alternative Medicine 2014, 14:203 - Effect of transcutaneous auricular vagus nerve stimulation on impaired glucose tolerance: a pilot randomized study.

3. Kozorosky, E. M., Lee, C.H., Lee, J. G., Nunez Martinez, V., Padayachee, L.E., & Stauss, H. M. (2022). Transcutaneous auricular vagus nerve stimulation augments postprandial inhibition of ghrelin. Physiological Reports, 10, e15253.

---

*本项目基于真实的科学研究数据，旨在为taVNS参数优化提供智能化解决方案。* 