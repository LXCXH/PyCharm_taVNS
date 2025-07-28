import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_processor import taVNSDataProcessor, create_data_loaders
from model import taVNSNet, ParamPredictionLoss, IndividualAdaptiveModule, ModelEvaluator, TFLiteCompatibilityChecker

class taVNSTrainer:
    """
    taVNS参数预测模型训练器
    包含完整的训练流程、验证、早停等功能
    增强版：支持TensorFlow Lite Micro部署
    """
    
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            min_lr=1e-6
        )
        
        # 损失函数
        self.criterion = ParamPredictionLoss(loss_type='mse')
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        # 最佳模型状态
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # 个体自适应模块（简化版）
        self.adaptive_module = IndividualAdaptiveModule(model, learning_rate=1e-4)
        
        # 创建输出目录
        # 确保Training_Outputs父目录存在
        parent_dir = "Training_Outputs"
        os.makedirs(parent_dir, exist_ok=True)
        
        # 在Training_Outputs下创建具体的训练输出目录
        self.output_dir = os.path.join(parent_dir, f"training_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_tflite_compatibility(self):
        """检查模型的TensorFlow Lite兼容性"""
        print("\n=== TensorFlow Lite兼容性检查 ===")
        compatibility_report = TFLiteCompatibilityChecker.check_model_compatibility(self.model)
        
        print(f"兼容性状态: {'✓ 兼容' if compatibility_report['compatible'] else '✗ 不兼容'}")
        
        if compatibility_report['issues']:
            print("发现的问题:")
            for issue in compatibility_report['issues']:
                print(f"  - {issue}")
        
        if compatibility_report['recommendations']:
            print("建议:")
            for rec in compatibility_report['recommendations']:
                print(f"  - {rec}")
        
        # 获取模型信息
        model_info = self.model.get_model_info()
        print(f"\n模型信息:")
        print(f"  - 总参数: {model_info['total_params']:,}")
        print(f"  - 可训练参数: {model_info['trainable_params']:,}")
        print(f"  - 输入形状: {model_info['input_shape']}")
        print(f"  - 输出形状: {model_info['output_shape']}")
        print(f"  - 架构类型: {model_info['architecture']}")
        print(f"  - TFLite兼容: {'是' if model_info['tflite_compatible'] else '否'}")
        
        return compatibility_report
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            input_glucose = batch['input_glucose'].to(self.device)
            target_params = batch['stim_params'].to(self.device)
            
            # 前向传播（简化调用，不使用individual_id）
            pred_params = self.model(input_glucose)
            
            # 计算损失
            loss = self.criterion(pred_params, target_params)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 返回平均损失
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取数据
                input_glucose = batch['input_glucose'].to(self.device)
                target_params = batch['stim_params'].to(self.device)
                
                # 前向传播（简化调用）
                pred_params = self.model(input_glucose)
                
                # 计算损失
                loss = self.criterion(pred_params, target_params)
                
                # 记录损失
                total_loss += loss.item()
                num_batches += 1
        
        # 返回平均损失
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=20):
        """
        完整训练流程
        """
        print(f"开始训练，共{epochs}个epoch...")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'data_processor_state': {
                        'glucose_scaler': self.data_processor.glucose_scaler,
                        'param_scaler': self.data_processor.param_scaler
                    }
                }, os.path.join(self.output_dir, 'best_model.pth'))
                
                print(f"  保存最佳模型 (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= early_stopping_patience:
                print(f"早停触发，在epoch {epoch+1}停止训练")
                break
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print("训练完成！")
        return self.train_losses, self.val_losses
    
    def plot_training_history(self):
        """
        绘制训练历史
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 总损失
        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0].set_title('Parameter Prediction Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 验证损失（放大显示）
        axes[1].plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
        axes[1].set_title('Validation Loss (Zoomed)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_config(self, config):
        """
        保存训练配置
        """
        config['training_completed'] = True
        config['best_val_loss'] = self.best_val_loss
        config['final_epoch'] = len(self.train_losses)
        
        # 添加TFLite兼容性信息
        compatibility_report = TFLiteCompatibilityChecker.check_model_compatibility(self.model)
        model_info = self.model.get_model_info()
        
        config['tflite_compatibility'] = {
            'compatible': compatibility_report['compatible'],
            'issues': compatibility_report['issues'],
            'recommendations': compatibility_report['recommendations'],
            'model_info': model_info
        }
        
        with open(os.path.join(self.output_dir, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def set_data_processor(self, data_processor):
        """
        设置数据处理器引用
        """
        self.data_processor = data_processor
    
    def optimize_for_tflite_deployment(self):
        """
        为TensorFlow Lite部署优化模型
        """
        print("\n=== 优化模型用于TensorFlow Lite部署 ===")
        
        # 应用TFLite优化
        optimized_model = TFLiteCompatibilityChecker.optimize_for_tflite(self.model)
        
        # 保存优化后的模型
        optimized_model_path = os.path.join(self.output_dir, 'tflite_optimized_model.pth')
        torch.save({
            'model_state_dict': optimized_model.state_dict(),
            'model_config': {
                'input_dim': optimized_model.input_dim,
                'param_dim': optimized_model.param_dim,
                'hidden_dim': optimized_model.hidden_dim
            },
            'optimization_applied': True,
            'tflite_ready': True
        }, optimized_model_path)
        
        print(f"TFLite优化模型已保存: {optimized_model_path}")
        
        return optimized_model

def main():
    """
    主训练函数
    """
    print("=== 基于三篇论文数据的taVNS参数预测模型训练 ===")
    print("=== TensorFlow Lite Micro兼容版本 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据处理器（启用数据扩充）
    print("创建数据处理器...")
    data_processor = taVNSDataProcessor(
        target_sequence_length=12, 
        enable_data_augmentation=True
    )
    
    # 生成基于论文的训练数据（包含大量扩充数据）
    print("生成基于论文的训练数据...")
    raw_samples = data_processor.create_comprehensive_dataset_from_papers()
    print(f"总共生成了 {len(raw_samples)} 个训练样本")
    
    # 显示论文数据总结
    paper_summary = data_processor.get_paper_summary()
    print("\n论文数据总结:")
    for paper, info in paper_summary.items():
        print(f"  {paper}:")
        print(f"    研究周期: {info['study_duration']}")
        print(f"    刺激参数: {info['stimulation']}")
        print(f"    效果: {info.get('glucose_reduction', info.get('effect', 'N/A'))}")
        print(f"    样本数: {info['sample_count']}")
    
    # 显示数据扩充总结
    augmentation_summary = data_processor.get_augmentation_summary()
    if isinstance(augmentation_summary, dict):
        print("\n数据扩充总结:")
        print(f"  扩充方法: {', '.join(augmentation_summary['augmentation_methods'])}")
        print(f"  噪声变体: {augmentation_summary['noise_levels']}")
        print(f"  时间扭曲变体: {augmentation_summary['time_warp_variants']}")
        print(f"  幅度变体: {augmentation_summary['amplitude_variants']}")
        print(f"  基线变体: {augmentation_summary['baseline_variants']}")
        print(f"  参数变异: {augmentation_summary['param_mutations']}")
        print(f"  个体差异: {augmentation_summary['individual_variations']}")
        print(f"  合成模式: {augmentation_summary['synthetic_patterns']}")
        print(f"  合成策略: {augmentation_summary['synthetic_strategies']}")
        print(f"  估计总样本: {augmentation_summary['estimated_total_samples']}")
    else:
        print(f"\n数据扩充状态: {augmentation_summary}")
    
    # 数据标准化
    print("\n数据标准化...")
    normalized_samples = data_processor.normalize_data(raw_samples)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        normalized_samples, 
        batch_size=32,  # 增加批次大小以处理更多数据
        train_ratio=0.8
    )
    
    # 创建TensorFlow Lite兼容模型
    print("创建TensorFlow Lite兼容模型...")
    model = taVNSNet(
        input_dim=12,
        param_dim=5,
        hidden_dim=128,  # 保持合理的隐藏层大小
        num_layers=2,
        dropout=0.2,
        num_individuals=100
    )
    
    # 创建训练器
    trainer = taVNSTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    # 设置数据处理器引用
    trainer.set_data_processor(data_processor)
    
    # 检查TensorFlow Lite兼容性
    compatibility_report = trainer.check_tflite_compatibility()
    
    # 训练配置
    config = {
        'model_config': {
            'input_dim': 12,
            'param_dim': 5,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'num_individuals': 100,
            'architecture_type': 'feedforward',
            'tflite_compatible': True
        },
        'training_config': {
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 20,
            'train_ratio': 0.8
        },
        'data_config': {
            'target_sequence_length': 12,
            'num_samples': len(raw_samples),
            'data_augmentation_enabled': True,
            'paper_summary': paper_summary,
            'augmentation_summary': augmentation_summary
        }
    }
    
    # 开始训练
    print("\n开始训练...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=20
    )
    
    # 绘制训练历史
    print("绘制训练历史...")
    trainer.plot_training_history()
    
    # 保存训练配置
    trainer.save_training_config(config)
    
    # 创建评估器
    evaluator = ModelEvaluator(data_processor)
    
    # 评估最终模型
    print("评估最终模型...")
    metrics, predictions = evaluator.evaluate_model(model, val_loader, device)
    
    print("\n=== 最终评估结果 ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # 保存评估结果 - 转换numpy类型为Python原生类型
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    with open(os.path.join(trainer.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # 保存数据处理器状态
    import pickle
    with open(os.path.join(trainer.output_dir, 'data_processor.pkl'), 'wb') as f:
        pickle.dump(data_processor, f)
    
    # 优化模型用于TensorFlow Lite部署
    optimized_model = trainer.optimize_for_tflite_deployment()
    
    print(f"\n训练完成！所有文件保存在: {trainer.output_dir}")
    print("主要文件:")
    print(f"  - 最佳模型: {trainer.output_dir}/best_model.pth")
    print(f"  - TFLite优化模型: {trainer.output_dir}/tflite_optimized_model.pth")
    print(f"  - 训练历史: {trainer.output_dir}/training_history.png")
    print(f"  - 训练配置: {trainer.output_dir}/training_config.json")
    print(f"  - 评估结果: {trainer.output_dir}/evaluation_results.json")
    print(f"  - 数据处理器: {trainer.output_dir}/data_processor.pkl")
    
    # 显示一些示例预测
    print("\n=== 示例预测 ===")
    model.eval()
    with torch.no_grad():
        # 从验证集中取一个批次
        sample_batch = next(iter(val_loader))
        input_glucose = sample_batch['input_glucose'][:3].to(device)
        target_params = sample_batch['stim_params'][:3].to(device)
        
        # 预测（简化调用）
        pred_params = model(input_glucose)
        
        # 反标准化
        input_glucose_orig = data_processor.inverse_transform_glucose(input_glucose.cpu().numpy())
        pred_params_orig = data_processor.inverse_transform_params(pred_params.cpu().numpy())
        target_params_orig = data_processor.inverse_transform_params(target_params.cpu().numpy())
        
        for i in range(3):
            print(f"\n样本 {i+1}:")
            print(f"  输入血糖: {input_glucose_orig[i].round(2)}")
            print(f"  预测参数: [频率={pred_params_orig[i][0]:.2f}Hz, 电流={pred_params_orig[i][1]:.2f}mA, "
                  f"时长={pred_params_orig[i][2]:.1f}min, 脉宽={pred_params_orig[i][3]:.0f}μs, 周期={pred_params_orig[i][4]:.1f}周]")
            print(f"  目标参数: [频率={target_params_orig[i][0]:.2f}Hz, 电流={target_params_orig[i][1]:.2f}mA, "
                  f"时长={target_params_orig[i][2]:.1f}min, 脉宽={target_params_orig[i][3]:.0f}μs, 周期={target_params_orig[i][4]:.1f}周]")
    
    print("\n=== TensorFlow Lite部署准备完成 ===")
    print("模型已优化并准备好转换为TensorFlow Lite格式")
    print("可以使用convert_to_tflite.py脚本进行最终转换")

if __name__ == "__main__":
    main() 