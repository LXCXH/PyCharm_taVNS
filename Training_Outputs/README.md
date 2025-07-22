# 训练输出文件夹

本文件夹统一存放所有taVNS模型的训练结果。

## 📁 文件夹说明

此文件夹包含：
- 历史训练结果（已整理移入）
- 新训练结果（运行`python train.py`时自动保存到此处）

## 🎯 自动保存

运行`python train.py`时，训练结果将自动保存到：
```
Training_Outputs/training_output_YYYYMMDD_HHMMSS/
```

## 📋 标准文件结构

每个训练结果文件夹包含：
- `best_model.pth` - 最佳模型权重
- `training_history.png` - 训练历史图表
- `evaluation_results.json` - 性能评估结果
- `training_config.json` - 训练配置参数
- `data_processor.pkl` - 数据处理器状态
- `checkpoint_epoch_*.pth` - 训练检查点

## 🚀 使用示例

```python
# 加载训练好的模型
import torch
from model import taVNSNet

model_path = 'Training_Outputs/training_output_xxx/best_model.pth'
model = taVNSNet()
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

*训练结果统一存储，便于管理和使用。* 