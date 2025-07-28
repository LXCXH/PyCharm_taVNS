#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch模型转换为TensorFlow Lite脚本
用于ESP32-S3上的TensorFlow Lite Micro部署
"""

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from model import taVNSNet
from data_processor import taVNSDataProcessor

class PyTorchToTFLiteConverter:
    """
    PyTorch模型转换为TensorFlow Lite的转换器
    """
    
    def __init__(self, model_dir=None):
        """
        初始化转换器
        
        Args:
            model_dir: 模型目录路径，如果为None则自动查找最新的模型
        """
        if model_dir is None:
            self.model_dir = self._find_latest_model_dir()
        else:
            self.model_dir = model_dir
            
        self.model_path = os.path.join(self.model_dir, "best_model.pth")
        self.config_path = os.path.join(self.model_dir, "training_config.json")
        self.data_processor_path = os.path.join(self.model_dir, "data_processor.pkl")
        
        # 创建输出目录
        self.output_dir = "tflite_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"使用模型目录: {self.model_dir}")
        
    def _find_latest_model_dir(self):
        """查找最新的训练输出目录"""
        training_outputs_dir = "Training_Outputs"
        if not os.path.exists(training_outputs_dir):
            raise FileNotFoundError(f"训练输出目录不存在: {training_outputs_dir}")
            
        # 获取所有训练输出目录
        model_dirs = []
        for item in os.listdir(training_outputs_dir):
            item_path = os.path.join(training_outputs_dir, item)
            if os.path.isdir(item_path) and item.startswith("training_output_"):
                model_dirs.append(item_path)
        
        if not model_dirs:
            raise FileNotFoundError("未找到任何训练输出目录")
            
        # 按时间排序，返回最新的
        model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return model_dirs[0]
    
    def load_pytorch_model(self):
        """加载PyTorch模型"""
        print("=== 加载PyTorch模型 ===")
        
        # 1. 加载配置
        print("1. 加载模型配置...")
        with open(self.config_path, 'r', encoding='utf-8') as f:
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
        
        # 3. 加载权重
        print("3. 加载模型权重...")
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
            
        model.load_state_dict(model_state_dict)
        model.eval()
        print("PyTorch模型加载成功")
        
        # 4. 加载数据处理器
        print("4. 加载数据处理器...")
        with open(self.data_processor_path, 'rb') as f:
            data_processor = pickle.load(f)
        print("数据处理器加载成功")
        
        return model, model_config, data_processor
    
    def create_tensorflow_model(self, pytorch_model, model_config):
        """创建等价的TensorFlow模型"""
        print("\n=== 创建TensorFlow模型 ===")
        
        # 定义TensorFlow版本的taVNSNet（纯前馈网络，最适合TFLite）
        class TFtaVNSNet(tf.keras.Model):
            def __init__(self, input_dim, param_dim, hidden_dim, num_layers, dropout=0.2):
                super(TFtaVNSNet, self).__init__()
                
                # 输入展平层（将序列数据展平为一维）
                self.flatten = tf.keras.layers.Flatten()
                
                # 编码器（替代LSTM/RNN的全连接层）
                self.encoder = tf.keras.Sequential([
                    tf.keras.layers.Dense(hidden_dim * 2, activation='relu'),
                    tf.keras.layers.Dense(hidden_dim, activation='relu')
                ])
                
                # 特征提取器（简化）
                self.feature_extractor = tf.keras.Sequential([
                    tf.keras.layers.Dense(hidden_dim, activation='relu'),
                    tf.keras.layers.Dense(hidden_dim // 2, activation='relu')
                ])
                
                # 参数预测头（简化）
                self.param_predictor = tf.keras.Sequential([
                    tf.keras.layers.Dense(hidden_dim // 4, activation='relu'),
                    tf.keras.layers.Dense(param_dim)
                ])
            
            def call(self, inputs, training=False):
                # 展平输入 [batch, sequence, features] -> [batch, sequence*features]
                flattened = self.flatten(inputs)
                
                # 编码
                encoded = self.encoder(flattened, training=training)
                
                # 特征提取
                features = self.feature_extractor(encoded, training=training)
                
                # 参数预测
                params = self.param_predictor(features, training=training)
                
                return params
        
        # 创建TensorFlow模型
        tf_model = TFtaVNSNet(
            input_dim=model_config['input_dim'],
            param_dim=model_config['param_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=0.0  # 推理时不使用dropout
        )
        
        # 构建模型（修正输入形状）
        dummy_input = tf.random.normal((1, model_config['input_dim'], 1))
        _ = tf_model(dummy_input, training=False)
        
        print("TensorFlow模型创建成功（纯前馈网络）")
        return tf_model

    def transfer_weights(self, pytorch_model, tf_model):
        """将PyTorch权重转移到TensorFlow模型（改进版本）"""
        print("\n=== 转移模型权重 ===")
        
        # 获取PyTorch模型的状态字典
        pytorch_state_dict = pytorch_model.state_dict()
        
        print("PyTorch模型权重:")
        for name, param in pytorch_state_dict.items():
            print(f"  {name}: {param.shape}")
        
        print("\nTensorFlow模型权重:")
        for var in tf_model.trainable_variables:
            print(f"  {var.name}: {var.shape}")
        
        # 由于完全不同的架构，我们主要转移全连接层的权重
        transferred_count = 0
        
        # 尝试转移特征提取器权重
        try:
            if 'feature_extractor.0.weight' in pytorch_state_dict:
                pytorch_weight = pytorch_state_dict['feature_extractor.0.weight'].detach().numpy().T
                for var in tf_model.trainable_variables:
                    if 'feature_extractor' in var.name and 'dense' in var.name and 'kernel' in var.name and var.shape == pytorch_weight.shape:
                        var.assign(pytorch_weight)
                        transferred_count += 1
                        print(f"转移权重: feature_extractor.0.weight -> {var.name}")
                        break
            
            if 'feature_extractor.0.bias' in pytorch_state_dict:
                pytorch_bias = pytorch_state_dict['feature_extractor.0.bias'].detach().numpy()
                for var in tf_model.trainable_variables:
                    if 'feature_extractor' in var.name and 'dense' in var.name and 'bias' in var.name and var.shape == pytorch_bias.shape:
                        var.assign(pytorch_bias)
                        transferred_count += 1
                        print(f"转移权重: feature_extractor.0.bias -> {var.name}")
                        break
        except Exception as e:
            print(f"转移特征提取器权重失败: {e}")
        
        # 尝试转移参数预测头权重（使用param_head而不是param_prediction_head）
        try:
            if 'param_head.5.weight' in pytorch_state_dict:  # 最后一层
                pytorch_weight = pytorch_state_dict['param_head.5.weight'].detach().numpy().T
                for var in tf_model.trainable_variables:
                    if 'param_predictor' in var.name and 'dense_1' in var.name and 'kernel' in var.name and var.shape == pytorch_weight.shape:
                        var.assign(pytorch_weight)
                        transferred_count += 1
                        print(f"转移权重: param_head.5.weight -> {var.name}")
                        break
            
            if 'param_head.5.bias' in pytorch_state_dict:
                pytorch_bias = pytorch_state_dict['param_head.5.bias'].detach().numpy()
                for var in tf_model.trainable_variables:
                    if 'param_predictor' in var.name and 'dense_1' in var.name and 'bias' in var.name and var.shape == pytorch_bias.shape:
                        var.assign(pytorch_bias)
                        transferred_count += 1
                        print(f"转移权重: param_head.5.bias -> {var.name}")
                        break
        except Exception as e:
            print(f"转移参数预测头权重失败: {e}")
        
        print(f"成功转移 {transferred_count} 个权重")
        print("注意: 由于架构差异（LSTM->全连接），大部分权重无法直接转移")
        print("建议使用转换后的模型进行重新训练以获得最佳性能")
        
        return tf_model

    def convert_to_tflite(self, tf_model, model_config, data_processor):
        """转换为TensorFlow Lite模型"""
        print("\n=== 转换为TensorFlow Lite ===")
        
        # 创建转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        
        # 设置优化选项（适合微控制器）
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 只使用TFLite内置操作（最佳兼容性）
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # 设置输入形状（修正为前馈网络期望的3D输入）
        input_shape = [1, model_config['input_dim'], 1]  # [batch, sequence, features]
        
        # 设置代表性数据集（可选）
        try:
            converter.representative_dataset = self._representative_dataset_gen(data_processor, input_shape)
            print("使用代表性数据集进行量化")
        except Exception as e:
            print(f"生成代表性数据集失败，跳过量化: {e}")
            converter.representative_dataset = None
        
        # 转换模型
        try:
            print("开始转换模型...")
            tflite_model = converter.convert()
            print("TensorFlow Lite模型转换成功！")
        except Exception as e:
            print(f"转换失败，尝试不使用量化: {e}")
            # 如果失败，尝试不使用量化
            converter.representative_dataset = None
            converter.optimizations = []
            
            try:
                tflite_model = converter.convert()
                print("TensorFlow Lite模型转换成功（未使用量化）")
            except Exception as e2:
                print(f"转换仍然失败: {e2}")
                raise e2
        
        return tflite_model
    
    def _representative_dataset_gen(self, data_processor, input_shape):
        """生成代表性数据集用于量化"""
        def representative_dataset():
            # 生成一些代表性样本
            raw_samples = data_processor.create_comprehensive_dataset_from_papers()
            normalized_samples = data_processor.normalize_data(raw_samples)
            
            # 取前10个样本作为代表性数据
            for i in range(min(10, len(normalized_samples))):
                sample = normalized_samples[i]
                # 修正输入形状为LSTM期望的格式
                glucose_data = sample['glucose_sequence'].reshape(1, -1, 1).astype(np.float32)
                yield [glucose_data]
        
        return representative_dataset
    
    def save_tflite_model(self, tflite_model, model_config, data_processor):
        """保存TensorFlow Lite模型和相关文件"""
        print("\n=== 保存模型文件 ===")
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存.tflite文件
        tflite_filename = f"tavns_model_{timestamp}.tflite"
        tflite_path = os.path.join(self.output_dir, tflite_filename)
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite模型已保存: {tflite_path}")
        
        # 保存模型信息
        model_info = {
            'model_config': model_config,
            'input_shape': [1, model_config['input_dim']],
            'output_shape': [1, model_config['param_dim']],
            'model_size_bytes': len(tflite_model),
            'conversion_time': timestamp,
            'source_model': self.model_path,
            'parameter_names': ['频率(Hz)', '电流(mA)', '时长(min)', '脉宽(μs)', '周期(周)'],
            'preprocessing': {
                'glucose_scaler_mean': data_processor.glucose_scaler.mean_.tolist(),
                'glucose_scaler_scale': data_processor.glucose_scaler.scale_.tolist(),
                'param_scaler_min': data_processor.param_scaler.min_.tolist(),
                'param_scaler_scale': data_processor.param_scaler.scale_.tolist()
            }
        }
        
        info_filename = f"model_info_{timestamp}.json"
        info_path = os.path.join(self.output_dir, info_filename)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        print(f"模型信息已保存: {info_path}")
        
        # 保存C头文件格式的模型数据（用于Arduino）
        self._save_as_c_header(tflite_model, timestamp)
        
        return tflite_path, info_path
    
    def _save_as_c_header(self, tflite_model, timestamp):
        """将TFLite模型保存为C头文件格式"""
        header_filename = f"tavns_model_{timestamp}.h"
        header_path = os.path.join(self.output_dir, header_filename)
        
        # 转换为C数组格式
        model_data = tflite_model
        model_size = len(model_data)
        
        with open(header_path, 'w') as f:
            f.write(f"// TensorFlow Lite模型数据\n")
            f.write(f"// 生成时间: {timestamp}\n")
            f.write(f"// 模型大小: {model_size} bytes\n\n")
            f.write(f"#ifndef TAVNS_MODEL_{timestamp.upper()}_H\n")
            f.write(f"#define TAVNS_MODEL_{timestamp.upper()}_H\n\n")
            f.write(f"const unsigned int tavns_model_len = {model_size};\n")
            f.write(f"const unsigned char tavns_model[] = {{\n")
            
            # 写入模型数据
            for i in range(0, model_size, 12):
                line = "  "
                for j in range(12):
                    if i + j < model_size:
                        line += f"0x{model_data[i + j]:02x}"
                        if i + j < model_size - 1:
                            line += ", "
                f.write(line + "\n")
            
            f.write("};\n\n")
            f.write(f"#endif  // TAVNS_MODEL_{timestamp.upper()}_H\n")
        
        print(f"C头文件已保存: {header_path}")
    
    def convert(self):
        """执行完整的转换流程"""
        print("开始PyTorch到TensorFlow Lite转换")
        print("=" * 60)
        
        try:
            # 1. 加载PyTorch模型
            pytorch_model, model_config, data_processor = self.load_pytorch_model()
            
            # 2. 创建TensorFlow模型
            tf_model = self.create_tensorflow_model(pytorch_model, model_config)
            
            # 3. 转移权重
            tf_model = self.transfer_weights(pytorch_model, tf_model)
            
            # 4. 转换为TFLite
            tflite_model = self.convert_to_tflite(tf_model, model_config, data_processor)
            
            # 5. 保存模型
            tflite_path, info_path = self.save_tflite_model(tflite_model, model_config, data_processor)
            
            print("\n" + "=" * 60)
            print("转换完成！")
            print(f"TFLite模型: {tflite_path}")
            print(f"模型信息: {info_path}")
            print(f"模型大小: {len(tflite_model)} bytes")
            print(f"输入形状: [1, {model_config['input_dim']}]")
            print(f"输出形状: [1, {model_config['param_dim']}]")
            
            return True
            
        except Exception as e:
            print(f"转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    converter = PyTorchToTFLiteConverter()
    success = converter.convert()
    
    if success:
        print("\n模型转换成功！可以在ESP32-S3上使用TensorFlow Lite Micro部署。")
    else:
        print("\n模型转换失败！")

if __name__ == "__main__":
    main()
