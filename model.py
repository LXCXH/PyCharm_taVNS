import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class taVNSNet(nn.Module):
    """
    taVNS血糖预测模型
    多任务学习架构：同时预测刺激参数和血糖序列
    支持个体自适应学习
    """
    
    def __init__(self, 
                 input_dim=12,           # 输入血糖序列长度
                 param_dim=5,            # 刺激参数维度
                 hidden_dim=128,         # 隐藏层维度
                 num_layers=2,           # LSTM层数
                 dropout=0.2,            # Dropout率
                 num_individuals=100):   # 最大个体数量
        super(taVNSNet, self).__init__()
        
        self.input_dim = input_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        
        # 血糖序列编码器 - LSTM
        self.glucose_encoder = nn.LSTM(
            input_size=1,                # 血糖值维度
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 个体嵌入层（用于适应不同个体）
        self.individual_embedding = nn.Embedding(num_individuals, hidden_dim // 4)
        
        # 刺激参数预测头
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, param_dim),
            nn.Sigmoid()  # 确保参数在[0,1]范围内
        )
        
        # 血糖预测头
        # combined_features = features + stim_params = (hidden_dim//2 + hidden_dim//4) + param_dim
        self.glucose_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4 + param_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, input_dim)
        )
        
        # 注意力机制（用于关注重要的时间点）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
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
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, glucose_seq, individual_id=None, return_attention=False):
        """
        前向传播
        Args:
            glucose_seq: 输入血糖序列 (batch_size, sequence_length)
            individual_id: 个体ID (batch_size,)
            return_attention: 是否返回注意力权重
        Returns:
            stim_params: 预测的刺激参数 (batch_size, param_dim)
            predicted_glucose: 预测的血糖序列 (batch_size, sequence_length)
            attention_weights: 注意力权重（可选）
        """
        batch_size = glucose_seq.size(0)
        
        # 添加通道维度
        glucose_seq = glucose_seq.unsqueeze(-1)  # (batch_size, sequence_length, 1)
        
        # LSTM编码血糖序列
        lstm_out, (hidden, cell) = self.glucose_encoder(glucose_seq)
        # lstm_out: (batch_size, sequence_length, hidden_dim)
        
        # 应用注意力机制
        attn_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # 取最后一个时间步的输出作为特征
        features = attn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 特征提取
        features = self.feature_extractor(features)  # (batch_size, hidden_dim//2)
        
        # 如果提供个体ID，添加个体嵌入
        if individual_id is not None:
            individual_emb = self.individual_embedding(individual_id)  # (batch_size, hidden_dim//4)
            features = torch.cat([features, individual_emb], dim=1)  # (batch_size, hidden_dim//2 + hidden_dim//4)
        else:
            # 如果没有个体ID，使用零向量
            individual_emb = torch.zeros(batch_size, self.hidden_dim // 4, device=glucose_seq.device)
            features = torch.cat([features, individual_emb], dim=1)
        
        # 预测刺激参数
        stim_params = self.param_head(features)  # (batch_size, param_dim)
        
        # 预测血糖序列
        combined_features = torch.cat([features, stim_params], dim=1)
        predicted_glucose = self.glucose_head(combined_features)  # (batch_size, input_dim)
        
        if return_attention:
            return stim_params, predicted_glucose, attention_weights
        else:
            return stim_params, predicted_glucose
    
    def get_individual_embedding(self, individual_id):
        """获取特定个体的嵌入向量"""
        return self.individual_embedding(individual_id)
    
    def update_individual_embedding(self, individual_id, new_embedding):
        """更新特定个体的嵌入向量（用于在线学习）"""
        with torch.no_grad():
            self.individual_embedding.weight[individual_id] = new_embedding

class AdaptiveLoss(nn.Module):
    """
    自适应损失函数
    结合刺激参数预测和血糖序列预测的损失
    """
    
    def __init__(self, param_weight=1.0, glucose_weight=2.0, 
                 param_loss_type='mse', glucose_loss_type='mse'):
        super(AdaptiveLoss, self).__init__()
        
        self.param_weight = param_weight
        self.glucose_weight = glucose_weight
        
        # 刺激参数损失
        if param_loss_type == 'mse':
            self.param_criterion = nn.MSELoss()
        elif param_loss_type == 'mae':
            self.param_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported param_loss_type: {param_loss_type}")
        
        # 血糖序列损失
        if glucose_loss_type == 'mse':
            self.glucose_criterion = nn.MSELoss()
        elif glucose_loss_type == 'mae':
            self.glucose_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported glucose_loss_type: {glucose_loss_type}")
    
    def forward(self, pred_params, target_params, pred_glucose, target_glucose):
        """
        计算总损失
        Args:
            pred_params: 预测的刺激参数
            target_params: 目标刺激参数
            pred_glucose: 预测的血糖序列
            target_glucose: 目标血糖序列
        Returns:
            total_loss: 总损失
            param_loss: 参数预测损失
            glucose_loss: 血糖预测损失
        """
        # 计算参数预测损失
        param_loss = self.param_criterion(pred_params, target_params)
        
        # 计算血糖序列预测损失
        glucose_loss = self.glucose_criterion(pred_glucose, target_glucose)
        
        # 总损失
        total_loss = self.param_weight * param_loss + self.glucose_weight * glucose_loss
        
        return total_loss, param_loss, glucose_loss

class IndividualAdaptiveModule:
    """
    个体自适应模块
    用于在线学习和个体化调整
    """
    
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.individual_optimizers = {}
        self.individual_histories = {}
    
    def create_individual_optimizer(self, individual_id):
        """为特定个体创建优化器"""
        if individual_id not in self.individual_optimizers:
            # 只优化该个体的嵌入向量
            optimizer = torch.optim.Adam(
                [self.model.individual_embedding.weight[individual_id]], 
                lr=self.learning_rate
            )
            self.individual_optimizers[individual_id] = optimizer
            self.individual_histories[individual_id] = []
    
    def adaptive_fine_tune(self, individual_id, input_glucose, target_params, 
                          target_glucose, epochs=5):
        """
        对特定个体进行微调
        Args:
            individual_id: 个体ID
            input_glucose: 输入血糖序列
            target_params: 目标刺激参数
            target_glucose: 目标血糖序列
            epochs: 微调轮数
        """
        # 创建个体优化器
        self.create_individual_optimizer(individual_id)
        optimizer = self.individual_optimizers[individual_id]
        
        # 损失函数
        criterion = AdaptiveLoss()
        
        # 微调
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            pred_params, pred_glucose = self.model(input_glucose, individual_id)
            
            # 计算损失
            total_loss, param_loss, glucose_loss = criterion(
                pred_params, target_params, pred_glucose, target_glucose
            )
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 记录历史
            self.individual_histories[individual_id].append({
                'epoch': epoch,
                'total_loss': total_loss.item(),
                'param_loss': param_loss.item(),
                'glucose_loss': glucose_loss.item()
            })
    
    def get_individual_performance(self, individual_id):
        """获取特定个体的性能历史"""
        return self.individual_histories.get(individual_id, [])
    
    def reset_individual(self, individual_id):
        """重置特定个体的学习历史"""
        if individual_id in self.individual_histories:
            self.individual_histories[individual_id] = []
        if individual_id in self.individual_optimizers:
            del self.individual_optimizers[individual_id]

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
        total_param_loss = 0
        total_glucose_loss = 0
        total_samples = 0
        
        param_predictions = []
        param_targets = []
        glucose_predictions = []
        glucose_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_glucose = batch['input_glucose'].to(device)
                target_params = batch['stim_params'].to(device)
                target_glucose = batch['output_glucose'].to(device)
                
                # 预测
                pred_params, pred_glucose = model(input_glucose)
                
                # 计算损失
                param_loss = F.mse_loss(pred_params, target_params)
                glucose_loss = F.mse_loss(pred_glucose, target_glucose)
                
                total_param_loss += param_loss.item() * input_glucose.size(0)
                total_glucose_loss += glucose_loss.item() * input_glucose.size(0)
                total_samples += input_glucose.size(0)
                
                # 收集预测结果
                param_predictions.append(pred_params.cpu().numpy())
                param_targets.append(target_params.cpu().numpy())
                glucose_predictions.append(pred_glucose.cpu().numpy())
                glucose_targets.append(target_glucose.cpu().numpy())
        
        # 计算平均损失
        avg_param_loss = total_param_loss / total_samples
        avg_glucose_loss = total_glucose_loss / total_samples
        
        # 反标准化预测结果
        param_predictions = np.concatenate(param_predictions, axis=0)
        param_targets = np.concatenate(param_targets, axis=0)
        glucose_predictions = np.concatenate(glucose_predictions, axis=0)
        glucose_targets = np.concatenate(glucose_targets, axis=0)
        
        # 反标准化
        param_predictions_orig = self.data_processor.inverse_transform_params(param_predictions)
        param_targets_orig = self.data_processor.inverse_transform_params(param_targets)
        glucose_predictions_orig = self.data_processor.inverse_transform_glucose(glucose_predictions)
        glucose_targets_orig = self.data_processor.inverse_transform_glucose(glucose_targets)
        
        # 计算评估指标
        metrics = {
            'param_mse': avg_param_loss,
            'glucose_mse': avg_glucose_loss,
            'param_mae': np.mean(np.abs(param_predictions_orig - param_targets_orig)),
            'glucose_mae': np.mean(np.abs(glucose_predictions_orig - glucose_targets_orig)),
            'param_rmse': np.sqrt(np.mean((param_predictions_orig - param_targets_orig) ** 2)),
            'glucose_rmse': np.sqrt(np.mean((glucose_predictions_orig - glucose_targets_orig) ** 2))
        }
        
        return metrics, {
            'param_predictions': param_predictions_orig,
            'param_targets': param_targets_orig,
            'glucose_predictions': glucose_predictions_orig,
            'glucose_targets': glucose_targets_orig
        } 