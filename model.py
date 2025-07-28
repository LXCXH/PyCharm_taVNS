import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class taVNSNet(nn.Module):
    """
    taVNS参数预测模型 - TensorFlow Lite Micro兼容版本
    使用纯前馈网络架构，适合ESP32-S3部署
    根据血糖序列预测最优的taVNS刺激参数
    """
    
    def __init__(self, 
                 input_dim=12,           # 输入血糖序列长度
                 param_dim=5,            # 刺激参数维度
                 hidden_dim=128,         # 隐藏层维度
                 num_layers=2,           # 网络层数（保留兼容性）
                 dropout=0.2,            # Dropout率（推理时不使用）
                 num_individuals=100):   # 最大个体数量（简化处理）
        super(taVNSNet, self).__init__()
        
        self.input_dim = input_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        
        # 血糖序列编码器 - 使用全连接层替代LSTM
        # 将序列数据展平后通过多层感知机处理
        self.glucose_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),  # 扩展维度
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # 特征提取网络 - 简化结构
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 个体适应层 - 简化为固定权重层
        # 不使用Embedding，而是通过额外的全连接层来模拟个体差异
        self.individual_adapter = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # taVNS参数预测头 - 简化结构
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, param_dim),
            nn.Sigmoid()  # 确保参数在[0,1]范围内
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, glucose_seq, individual_id=None, return_attention=False):
        """
        前向传播 - 简化版本，适合TFLite转换
        Args:
            glucose_seq: 输入血糖序列 (batch_size, sequence_length)
            individual_id: 个体ID（简化处理，不使用）
            return_attention: 是否返回注意力权重（不支持）
        Returns:
            stim_params: 预测的刺激参数 (batch_size, param_dim)
        """
        batch_size = glucose_seq.size(0)
        
        # 确保输入是2D张量 [batch_size, sequence_length]
        if glucose_seq.dim() == 3:
            glucose_seq = glucose_seq.squeeze(-1)
        
        # 血糖序列编码 - 直接处理展平的序列
        encoded_features = self.glucose_encoder(glucose_seq)
        
        # 特征提取
        features = self.feature_extractor(encoded_features)
        
        # 个体适应（简化处理，不依赖individual_id）
        adapted_features = self.individual_adapter(features)
        
        # 预测taVNS刺激参数
        stim_params = self.param_head(adapted_features)
        
        # 简化返回，不支持attention
        return stim_params
    
    def get_model_info(self):
        """获取模型信息，用于TFLite转换"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_shape': [1, self.input_dim],
            'output_shape': [1, self.param_dim],
            'architecture': 'feedforward',
            'tflite_compatible': True
        }

class ParamPredictionLoss(nn.Module):
    """
    taVNS参数预测损失函数
    """
    
    def __init__(self, loss_type='mse'):
        super(ParamPredictionLoss, self).__init__()
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    def forward(self, pred_params, target_params):
        """
        计算参数预测损失
        Args:
            pred_params: 预测的刺激参数
            target_params: 目标刺激参数
        Returns:
            loss: 损失值
        """
        return self.criterion(pred_params, target_params)

class IndividualAdaptiveModule:
    """
    个体自适应模块 - 简化版本
    用于在线学习和个体化调整
    """
    
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.individual_histories = {}
        self.adaptation_enabled = False  # TFLite部署时禁用
    
    def adaptive_fine_tune(self, individual_id, input_glucose, target_params, epochs=5):
        """
        对特定个体进行微调 - 简化版本
        """
        if not self.adaptation_enabled:
            print("个体自适应在TFLite部署模式下已禁用")
            return
        
        # 简化的微调逻辑
        criterion = ParamPredictionLoss()
        
        for epoch in range(epochs):
            pred_params = self.model(input_glucose)
            loss = criterion(pred_params, target_params)
            
            # 记录历史
            if individual_id not in self.individual_histories:
                self.individual_histories[individual_id] = []
            
            self.individual_histories[individual_id].append({
                'epoch': epoch,
                'loss': loss.item()
            })
    
    def get_individual_performance(self, individual_id):
        """获取特定个体的性能历史"""
        return self.individual_histories.get(individual_id, [])
    
    def reset_individual(self, individual_id):
        """重置特定个体的学习历史"""
        if individual_id in self.individual_histories:
            self.individual_histories[individual_id] = []

class ModelEvaluator:
    """
    模型评估器
    提供各种评估指标
    """
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def evaluate_model(self, model, dataloader, device):
        """
        评估模型性能
        """
        model.eval()
        total_loss = 0
        total_samples = 0
        
        param_predictions = []
        param_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_glucose = batch['input_glucose'].to(device)
                target_params = batch['stim_params'].to(device)
                
                # 预测
                pred_params = model(input_glucose)
                
                # 计算损失
                loss = F.mse_loss(pred_params, target_params)
                
                total_loss += loss.item() * input_glucose.size(0)
                total_samples += input_glucose.size(0)
                
                # 收集预测结果
                param_predictions.append(pred_params.cpu().numpy())
                param_targets.append(target_params.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / total_samples
        
        # 反标准化预测结果
        param_predictions = np.concatenate(param_predictions, axis=0)
        param_targets = np.concatenate(param_targets, axis=0)
        
        # 反标准化
        param_predictions_orig = self.data_processor.inverse_transform_params(param_predictions)
        param_targets_orig = self.data_processor.inverse_transform_params(param_targets)
        
        # 计算评估指标
        metrics = {
            'param_mse': avg_loss,
            'param_mae': np.mean(np.abs(param_predictions_orig - param_targets_orig)),
            'param_rmse': np.sqrt(np.mean((param_predictions_orig - param_targets_orig) ** 2))
        }
        
        return metrics, {
            'param_predictions': param_predictions_orig,
            'param_targets': param_targets_orig
        }

class TFLiteCompatibilityChecker:
    """
    TensorFlow Lite兼容性检查器
    """
    
    @staticmethod
    def check_model_compatibility(model):
        """
        检查模型是否兼容TensorFlow Lite
        """
        compatibility_report = {
            'compatible': True,
            'issues': [],
            'recommendations': []
        }
        
        # 检查是否使用了不兼容的层
        incompatible_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                incompatible_layers.append(f"RNN层: {name}")
            elif isinstance(module, nn.MultiheadAttention):
                incompatible_layers.append(f"注意力层: {name}")
            elif isinstance(module, nn.Embedding):
                incompatible_layers.append(f"嵌入层: {name}")
        
        if incompatible_layers:
            compatibility_report['compatible'] = False
            compatibility_report['issues'].extend(incompatible_layers)
            compatibility_report['recommendations'].append("使用全连接层替代复杂结构")
        
        # 检查模型大小
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 1000000:  # 1M参数
            compatibility_report['recommendations'].append("模型参数过多，建议减少隐藏层维度")
        
        return compatibility_report
    
    @staticmethod
    def optimize_for_tflite(model):
        """
        为TensorFlow Lite优化模型
        """
        # 设置为评估模式
        model.eval()
        
        # 禁用所有dropout
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        
        return model 