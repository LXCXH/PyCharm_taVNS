#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的taVNS模型测试脚本
用于验证模型加载和基本预测功能
"""

import torch
import numpy as np
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from model import taVNSNet
from data_processor import taVNSDataProcessor

def test_model_loading():
    """测试模型加载"""
    print("=== 测试模型加载 ===")
    
    # 最新模型路径
    latest_model_dir = "Training_Outputs/training_output_20250728_143601"
    model_path = os.path.join(latest_model_dir, "best_model.pth")
    data_processor_path = os.path.join(latest_model_dir, "data_processor.pkl")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    if not os.path.exists(data_processor_path):
        print(f"错误: 数据处理器文件不存在: {data_processor_path}")
        return False
    
    try:
        # 1. 加载模型配置
        print("1. 加载模型配置...")
        config_path = os.path.join(latest_model_dir, 'training_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model_config = config['model_config']
        print(f"模型配置: {model_config}")
        
        # 2. 初始化模型
        print("2. 初始化模型...")
        device = torch.device('cpu')
        model = taVNSNet(
            input_dim=model_config['input_dim'],
            param_dim=model_config['param_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            num_individuals=model_config['num_individuals']
        )
        print("模型初始化成功")
        
        # 3. 加载模型权重
        print("3. 加载模型权重...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 检查是否是完整的检查点文件
        if 'model_state_dict' in checkpoint:
            print("检测到完整检查点文件，提取模型状态字典...")
            model_state_dict = checkpoint['model_state_dict']
        else:
            print("直接使用加载的权重...")
            model_state_dict = checkpoint
            
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        print("模型权重加载成功")
        
        # 4. 加载数据处理器
        print("4. 加载数据处理器...")
        with open(data_processor_path, 'rb') as f:
            data_processor = pickle.load(f)
        print("数据处理器加载成功")
        
        return True, model, data_processor, device
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def test_prediction(model, data_processor, device):
    """测试预测功能"""
    print("\n=== 测试预测功能 ===")
    
    try:
        # 设置随机种子，确保结果可重现
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 从数据处理器中获取训练时的验证数据
        from data_processor import create_data_loaders
        
        # 重新生成训练数据
        raw_samples = data_processor.create_comprehensive_dataset_from_papers()
        normalized_samples = data_processor.normalize_data(raw_samples)
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            normalized_samples, 
            batch_size=16, 
            train_ratio=0.8
        )
        
        # 从验证集中取一个批次，模拟训练时的示例预测
        model.eval()  # 确保模型在评估模式
        with torch.no_grad():
            # 从验证集中取一个批次
            sample_batch = next(iter(val_loader))
            input_glucose = sample_batch['input_glucose'][:3].to(device)
            target_params = sample_batch['stim_params'][:3].to(device)
            
            # 预测 - 不使用individual_id，与训练时保持一致
            pred_params = model(input_glucose)
            
            # 反标准化
            input_glucose_orig = data_processor.inverse_transform_glucose(input_glucose.cpu().numpy())
            pred_params_orig = data_processor.inverse_transform_params(pred_params.cpu().numpy())
            target_params_orig = data_processor.inverse_transform_params(target_params.cpu().numpy())
            
            for i in range(3):
                print(f"\n--- 样本 {i+1} ---")
                print(f"输入血糖序列: {input_glucose_orig[i].tolist()}")
                
                # 格式化参数，保持小数点对齐
                target_formatted = [f"{x:.2f}" for x in target_params_orig[i]]
                predicted_formatted = [f"{x:.2f}" for x in pred_params_orig[i]]
                
                print(f"目标参数: {target_formatted}")
                print(f"预测参数: {predicted_formatted}")
        
        print("\n预测功能测试成功！")
        return True
        
    except Exception as e:
        print(f"预测功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("taVNS模型简单测试")
    print("=" * 50)
    
    # 测试模型加载
    success, model, data_processor, device = test_model_loading()
    
    if success:
        print("\n模型加载成功！")
        
        # 测试预测功能
        prediction_success = test_prediction(model, data_processor, device)
        
        if prediction_success:
            print("\n所有测试通过！模型可以正常使用。")
        else:
            print("\n预测功能测试失败。")
    else:
        print("\n模型加载失败。")

if __name__ == "__main__":
    main() 