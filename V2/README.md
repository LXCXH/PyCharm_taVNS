# taVNS血糖预测模型 - 基于三篇论文数据

基于三篇真实论文数据的经皮耳迷走神经刺激（taVNS）血糖预测模型，能够根据输入的血糖序列预测最优的刺激参数和刺激后的血糖变化。

## 项目概述

### 模型目标
- **输入**: 12个血糖值的时间序列（每5分钟一个，共1小时）
- **输出**: 
  - 推荐的taVNS刺激参数（频率、电流、持续时间、脉宽、治疗周期）
  - 预测的刺激后血糖序列

### 核心特性
- **多任务学习**: 同时预测刺激参数和血糖序列
- **个体自适应**: 支持不同个体的个性化调整
- **注意力机制**: 关注重要的时间点
- **基于真实论文数据**: 所有参数和效应都基于三篇真实的taVNS研究论文

## 数据来源

### 论文1: ZDF小鼠实验
- **研究类型**: 动物实验（ZDF糖尿病小鼠）
- **刺激参数**: 2/15 Hz, 2mA, 30分钟
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
V2/
├── data_processor.py      # 基于论文数据的数据处理模块
├── model.py              # 神经网络模型定义
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── requirements.txt      # 依赖包
├── README.md            # 项目说明
└── 论文相关文件/
    ├── 论文参数总结_V2.xlsx
    ├── 论文参数总结_V2.docx
    └── 三篇PDF论文
```

## 数据处理详解

### 数据生成策略

#### 1. ZDF小鼠数据处理
```python
# 长期效应数据（5周）
control_glucose = [18, 20, 22, 24, 26, 28]  # 对照组
treatment_glucose = [18, 10, 10, 10, 10, 10]  # 治疗组

# 胰腺切除实验数据
pancreatic_removal_glucose = {
    'day1': [25, 24.5, 26.5, 29.5, 27.5, 26, 24.5],
    'day3': [22, 24, 29, 27, 24, 23.5, 23.5],
    'day5': [21, 24, 30, 24.5, 22, 21, 19.5]
}
```

#### 2. 人体IGT患者数据处理
```python
# 2小时血糖耐量测试结果
results = {
    'baseline': {'ta_vns': 9.7, 'sham': 9.1, 'no_treatment': 9.3},
    '6_weeks': {'ta_vns': 7.3, 'sham': 8.0, 'no_treatment': 9.5},
    '12_weeks': {'ta_vns': 7.5, 'sham': 8.0, 'no_treatment': 10.0}
}
```

#### 3. 健康人餐后血糖数据处理
```python
# 两种协议的血糖响应（mg/dL转换为mmol/L）
protocol1_sham = [105, 110, 115, 120, 115, 110, 105, 100, 95, 90, 85, 80]
protocol1_tavns = [100, 105, 110, 115, 110, 105, 100, 95, 90, 85, 80, 75]
```

### 生理模型

#### 刺激强度计算
```python
def calculate_stimulation_intensity(stim_params):
    freq, amp, duration, pulse_width, session_duration = stim_params
    base_intensity = (freq * amp * duration * pulse_width) / 1000000
    
    # 考虑治疗周期的累积效应
    if session_duration >= 6:
        base_intensity *= 1.3  # 长期治疗增强效应
    elif session_duration >= 2:
        base_intensity *= 1.1
    
    return base_intensity
```

#### 血糖变化模拟
```python
def simulate_tavns_effect(glucose_sequence, stim_params, sensitivity):
    intensity = calculate_stimulation_intensity(stim_params)
    reduction_factor = intensity * 0.1 * sensitivity
    
    # 时间依赖的效应
    time_weights = np.linspace(0.5, 1.0, 12)
    glucose_changes = reduction_factor * time_weights * (glucose_sequence / 10.0)
    
    return glucose_sequence - glucose_changes
```

## 模型架构

### 核心模型：taVNSNet

**网络结构**:
1. **LSTM编码器**: 处理血糖时间序列
2. **多头注意力**: 关注重要时间点
3. **特征提取器**: 提取高级特征
4. **个体嵌入层**: 适应不同个体
5. **双任务头**: 
   - 参数预测头：输出刺激参数
   - 血糖预测头：输出预测血糖序列

**参数维度**:
- 输入维度: 12 (血糖序列长度)
- 参数维度: 5 (频率、电流、持续时间、脉宽、治疗周期)
- 隐藏维度: 128
- LSTM层数: 2
- 注意力头数: 4

### 损失函数
```python
total_loss = param_weight × param_loss + glucose_weight × glucose_loss
# 默认权重: param_weight=1.0, glucose_weight=2.0
```

## 使用方法

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
cd V2
python train.py
```

**训练输出**:
- 生成基于三篇论文的训练数据
- 显示论文数据总结
- 训练模型并保存最佳权重
- 生成训练历史图表

### 3. 测试模型
```bash
python test.py
```

### 4. 交互式预测
```python
from test import taVNSTester

# 加载模型
tester = taVNSTester('training_output_xxx/best_model.pth')

# 预测示例
glucose_sequence = [8.0, 8.5, 9.2, 9.8, 9.5, 9.0, 8.5, 8.2, 8.0, 7.8, 7.6, 7.4]
pred_params, pred_glucose = tester.predict_single_sample(glucose_sequence)

print("推荐刺激参数:", pred_params)
print("预测血糖序列:", pred_glucose)
```

## 训练数据统计

基于三篇论文生成的训练样本统计：

| 论文来源 | 研究类型 | 样本数量 | 主要特点 |
|----------|----------|----------|----------|
| 论文1 | ZDF小鼠 | ~21个 | 长期效应(5周) + 胰腺切除实验 |
| 论文2 | 人体IGT | ~15个 | 12周临床试验，包含对照组 |
| 论文3 | 健康人 | ~10个 | 急性效应，两种刺激协议 |
| **总计** | **多类型** | **~46个基础样本** | **通过个体差异扩展到数百个** |

## 个体差异建模

### 敏感性参数
- **ZDF小鼠**: 0.8, 1.0, 1.2
- **IGT患者**: 0.6, 0.8, 1.0, 1.2, 1.4
- **健康人**: 0.7, 0.9, 1.0, 1.1, 1.3

### 个体自适应机制
- 每个个体有独特的嵌入向量
- 在线学习更新个体特征
- 只更新个体嵌入，保持主模型稳定

## 评估指标

### 性能指标
- **参数预测**: MSE, MAE, RMSE
- **血糖预测**: MSE, MAE, RMSE
- **个体自适应**: 性能改进百分比

### 预期性能
- 参数预测误差: < 15%
- 血糖预测误差: < 10%
- 个体自适应改进: > 20%

## 实际应用场景

### 1. 临床决策支持
- 根据患者血糖模式推荐个性化刺激参数
- 预测治疗效果
- 优化治疗方案

### 2. 研究工具
- 探索不同刺激参数的效应
- 分析个体差异
- 设计新的实验方案

### 3. 设备优化
- 智能调节刺激参数
- 实时血糖监测与反馈
- 自动化治疗系统

## 模型优势

### 1. 科学可靠性
- 基于真实论文数据
- 涵盖动物实验和人体试验
- 包含多种实验设计

### 2. 技术先进性
- 多任务学习架构
- 注意力机制
- 个体自适应能力

### 3. 实用性
- 端到端预测
- 快速推理
- 易于集成

## 注意事项

1. **数据来源**: 所有数据都基于已发表的科学论文
2. **模型限制**: 当前模型基于有限的论文数据，实际应用需要更多验证
3. **临床应用**: 本模型仅供研究使用，临床应用需要监管部门批准
4. **个体差异**: 模型考虑了个体差异，但实际应用中仍需个体化调整

## 未来改进方向

1. **更多数据源**: 整合更多论文和临床数据
2. **高级模型**: 使用Transformer等更先进的架构
3. **多模态输入**: 结合其他生理信号
4. **实时优化**: 开发在线学习算法
5. **临床验证**: 进行前瞻性临床试验

## 参考文献

1. Wang S, Zhai X, Li S, McCabe MF, Wang X, Rong P (2015) Transcutaneous Vagus Nerve Stimulation Induces Tidal Melatonin Secretion and Has an Antidiabetic Effect in Zucker Fatty Rats. PLoS ONE 10(4): e0124195.

2. Huang et al. BMC Complementary and Alternative Medicine 2014, 14:203 - Effect of transcutaneous auricular vagus nerve stimulation on impaired glucose tolerance: a pilot randomized study.

3. Kozorosky, E. M., Lee, C.H., Lee, J. G., Nunez Martinez, V., Padayachee, L.E., & Stauss, H. M. (2022). Transcutaneous auricular vagus nerve stimulation augments postprandial inhibition of ghrelin. Physiological Reports, 10, e15253.

## 联系方式

如有问题或建议，请联系项目维护者。

---

*本项目基于真实的科学研究数据，旨在为taVNS血糖调节提供智能化解决方案。* 