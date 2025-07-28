#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch模型转换为TensorFlow Lite脚本
用于ESP32-S3上的TensorFlow Lite Micro部署
包含转换后的模型测试和预测对比
改进版：提高转换准确度
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
    改进版：提高转换准确度
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
        
        # 创建主输出目录
        self.base_output_dir = "TFLite_Output"
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # 为本次转换创建专用文件夹
        self.conversion_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(self.base_output_dir, f"conversion_{self.conversion_timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用模型目录: {self.model_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"使用设备: {self.device}")
    
    def _find_latest_model_dir(self):
        """查找最新的模型目录"""
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
        model_dirs.sort(reverse=True)
        return model_dirs[0]
    
    def load_model_config(self):
        """加载模型配置"""
        print("\n=== 加载模型配置 ===")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        model_config = config['model_config']
        print(f"模型配置: {model_config}")
        
        return model_config
    
    def load_pytorch_model(self, model_config):
        """加载PyTorch模型"""
        print("\n=== 加载PyTorch模型 ===")
        
        # 创建模型实例
        model = taVNSNet(
            input_dim=model_config['input_dim'],
            param_dim=model_config['param_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.2),
            num_individuals=model_config.get('num_individuals', 100)
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 检查是否是完整的检查点文件
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
            
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        
        print("PyTorch模型加载成功")
        
        # 打印模型信息和权重统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {total_params:,}")
        
        # 打印权重统计信息用于调试
        print("\nPyTorch模型权重统计:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape} | mean: {param.mean().item():.6f} | std: {param.std().item():.6f}")
        
        return model
    
    def load_data_processor(self):
        """加载数据处理器"""
        print("\n=== 加载数据处理器 ===")
        
        with open(self.data_processor_path, 'rb') as f:
            data_processor = pickle.load(f)
        
        print("数据处理器加载成功")
        return data_processor
    
    def create_tensorflow_model(self, pytorch_model, model_config):
        """创建等价的TensorFlow模型 - 改进版"""
        print("\n=== 创建TensorFlow模型 ===")
        
        # 使用函数式API创建更精确的模型
        input_layer = tf.keras.layers.Input(shape=(model_config['input_dim'],), name='input')
        
        # 血糖序列编码器 - 使用与PyTorch完全相同的结构
        x = tf.keras.layers.Dense(
            model_config['hidden_dim'] * 2, 
            activation='relu', 
            name='glucose_encoder_1',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )(input_layer)
        
        x = tf.keras.layers.Dense(
            model_config['hidden_dim'], 
            activation='relu', 
            name='glucose_encoder_2',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )(x)
        
        # 特征提取器
        x = tf.keras.layers.Dense(
            model_config['hidden_dim'], 
            activation='relu', 
            name='feature_extractor_1',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )(x)
        
        x = tf.keras.layers.Dense(
            model_config['hidden_dim'] // 2, 
            activation='relu', 
            name='feature_extractor_2',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )(x)
        
        # 个体适应层
        x = tf.keras.layers.Dense(
            model_config['hidden_dim'] // 4, 
            activation='relu', 
            name='individual_adapter_1',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )(x)
        
        # 参数预测头
        x = tf.keras.layers.Dense(
            model_config['hidden_dim'] // 4, 
            activation='relu', 
            name='param_head_1',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )(x)
        
        x = tf.keras.layers.Dense(
            model_config['hidden_dim'] // 8, 
            activation='relu', 
            name='param_head_2',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )(x)
        
        output = tf.keras.layers.Dense(
            model_config['param_dim'], 
            activation='sigmoid', 
            name='param_output',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )(x)
        
        # 创建模型
        tf_model = tf.keras.Model(inputs=input_layer, outputs=output, name='taVNSNet_TF')
        
        print("TensorFlow模型创建成功")
        print(f"TensorFlow模型结构:")
        tf_model.summary()
        
        return tf_model
    
    def transfer_weights_improved(self, pytorch_model, tf_model):
        """改进的权重转移方法"""
        print("\n=== 改进的权重转移 ===")
        
        # 获取PyTorch模型的状态字典
        pytorch_state_dict = pytorch_model.state_dict()
        
        # 改进的权重映射 - 使用层名称而不是变量名
        layer_mapping = {
            # 编码器层
            ('glucose_encoder.0.weight', 'glucose_encoder.0.bias'): 'glucose_encoder_1',
            ('glucose_encoder.2.weight', 'glucose_encoder.2.bias'): 'glucose_encoder_2',
            
            # 特征提取器层
            ('feature_extractor.0.weight', 'feature_extractor.0.bias'): 'feature_extractor_1',
            ('feature_extractor.2.weight', 'feature_extractor.2.bias'): 'feature_extractor_2',
            
            # 个体适应层
            ('individual_adapter.0.weight', 'individual_adapter.0.bias'): 'individual_adapter_1',
            
            # 参数预测头
            ('param_head.0.weight', 'param_head.0.bias'): 'param_head_1',
            ('param_head.2.weight', 'param_head.2.bias'): 'param_head_2',
            ('param_head.4.weight', 'param_head.4.bias'): 'param_output'
        }
        
        transferred_count = 0
        total_layers = len(layer_mapping)
        
        for (pytorch_weight_name, pytorch_bias_name), tf_layer_name in layer_mapping.items():
            # 获取TensorFlow层
            tf_layer = tf_model.get_layer(tf_layer_name)
            
            # 转移权重
            if pytorch_weight_name in pytorch_state_dict and pytorch_bias_name in pytorch_state_dict:
                pytorch_weight = pytorch_state_dict[pytorch_weight_name].cpu().numpy()
                pytorch_bias = pytorch_state_dict[pytorch_bias_name].cpu().numpy()
                
                # PyTorch权重是 [out_features, in_features]，需要转置为 [in_features, out_features]
                pytorch_weight_transposed = pytorch_weight.T
                
                # 检查形状
                tf_weights = tf_layer.get_weights()
                if len(tf_weights) == 2:  # 权重和偏置
                    tf_weight_shape = tf_weights[0].shape
                    tf_bias_shape = tf_weights[1].shape
                    
                    if pytorch_weight_transposed.shape == tf_weight_shape and pytorch_bias.shape == tf_bias_shape:
                        # 设置权重
                        tf_layer.set_weights([pytorch_weight_transposed, pytorch_bias])
                        transferred_count += 1
                        print(f"✓ 成功转移: {tf_layer_name}")
                        print(f"  权重形状: {pytorch_weight_transposed.shape}")
                        print(f"  偏置形状: {pytorch_bias.shape}")
                        print(f"  权重统计: mean={pytorch_weight_transposed.mean():.6f}, std={pytorch_weight_transposed.std():.6f}")
                    else:
                        print(f"✗ 形状不匹配: {tf_layer_name}")
                        print(f"  PyTorch权重: {pytorch_weight_transposed.shape} vs TF权重: {tf_weight_shape}")
                        print(f"  PyTorch偏置: {pytorch_bias.shape} vs TF偏置: {tf_bias_shape}")
                else:
                    print(f"✗ TensorFlow层权重数量异常: {tf_layer_name}")
            else:
                print(f"✗ 未找到PyTorch权重: {pytorch_weight_name} 或 {pytorch_bias_name}")
        
        print(f"\n权重转移完成: {transferred_count}/{total_layers} 层")
        
        # 验证权重转移
        if transferred_count == total_layers:
            print("✓ 所有权重转移成功")
            self._verify_weight_transfer(pytorch_model, tf_model)
        else:
            print("⚠ 部分权重转移失败")
        
        return transferred_count == total_layers
    
    def _verify_weight_transfer(self, pytorch_model, tf_model):
        """验证权重转移的正确性"""
        print("\n=== 验证权重转移 ===")
        
        # 生成随机测试输入
        test_input = np.random.randn(1, 12).astype(np.float32)
        
        # PyTorch预测
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_input = torch.FloatTensor(test_input).to(self.device)
            pytorch_output = pytorch_model(pytorch_input).cpu().numpy()
        
        # TensorFlow预测
        tf_output = tf_model(test_input).numpy()
        
        # 计算差异
        max_diff = np.max(np.abs(pytorch_output - tf_output))
        mean_diff = np.mean(np.abs(pytorch_output - tf_output))
        
        print(f"PyTorch输出: {pytorch_output[0]}")
        print(f"TensorFlow输出: {tf_output[0]}")
        print(f"最大差异: {max_diff:.8f}")
        print(f"平均差异: {mean_diff:.8f}")
        
        if max_diff < 1e-5:
            print("✓ 权重转移验证通过 (差异 < 1e-5)")
            return True
        elif max_diff < 1e-3:
            print("⚠ 权重转移基本正确 (差异 < 1e-3)")
            return True
        else:
            print("✗ 权重转移可能有问题 (差异较大)")
            return False
    
    def convert_to_tflite(self, tf_model, model_config):
        """转换为TensorFlow Lite模型 - 改进版"""
        print("\n=== 转换为TensorFlow Lite ===")
        
        # 创建转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        
        # 设置优化选项 - 使用更保守的优化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 只使用内置操作（适合微控制器）
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # 设置输入形状
        input_shape = [1, model_config['input_dim']]
        
        # 添加代表性数据集以提高量化质量
        def representative_dataset():
            for _ in range(100):
                # 生成代表性输入数据
                data = np.random.randn(1, model_config['input_dim']).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        
        try:
            # 转换模型
            tflite_model = converter.convert()
            print("TensorFlow Lite转换成功")
            
            # 保存模型 - 使用统一的时间戳
            tflite_filename = f"tavns_model_improved.tflite"
            tflite_path = os.path.join(self.output_dir, tflite_filename)
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TFLite模型已保存: {tflite_path}")
            
            # 保存模型信息
            model_info = {
                'model_path': tflite_path,
                'input_shape': input_shape,
                'output_shape': [1, model_config['param_dim']],
                'model_size_bytes': len(tflite_model),
                'conversion_time': self.conversion_timestamp,
                'original_model_dir': self.model_dir,
                'improvements': [
                    'improved_weight_transfer',
                    'weight_transfer_verification', 
                    'representative_dataset',
                    'conservative_optimization'
                ]
            }
            
            info_path = os.path.join(self.output_dir, f"model_info.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"模型信息已保存: {info_path}")
            print(f"模型大小: {len(tflite_model):,} 字节 ({len(tflite_model)/1024:.1f} KB)")
            
            return tflite_path, model_info
            
        except Exception as e:
            print(f"TensorFlow Lite转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def test_model_comparison(self, pytorch_model, tflite_path, data_processor, model_config):
        """测试PyTorch模型和TFLite模型的预测对比 - 改进版"""
        print("\n=== 模型预测对比测试 ===")
        
        if tflite_path is None:
            print("TFLite模型不存在，跳过对比测试")
            return
        
        # 加载TFLite模型
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # 获取输入输出详情
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"TFLite输入形状: {input_details[0]['shape']}")
        print(f"TFLite输出形状: {output_details[0]['shape']}")
        print(f"TFLite输入类型: {input_details[0]['dtype']}")
        print(f"TFLite输出类型: {output_details[0]['dtype']}")
        
        # 生成测试样本
        test_samples = self._generate_test_samples(data_processor)
        
        print(f"\n开始对比测试 ({len(test_samples)} 个样本)...")
        
        pytorch_predictions = []
        tflite_predictions = []
        mae_errors = []
        normalized_mae_errors = []  # 标准化空间的误差
        
        for i, (glucose_seq, description) in enumerate(test_samples):
            # PyTorch预测
            pytorch_model.eval()
            with torch.no_grad():
                glucose_tensor = torch.FloatTensor(glucose_seq).unsqueeze(0).to(self.device)
                pytorch_pred_norm = pytorch_model(glucose_tensor).cpu().numpy()[0]
            
            # TFLite预测
            glucose_input = glucose_seq.astype(np.float32).reshape(1, -1)
            interpreter.set_tensor(input_details[0]['index'], glucose_input)
            interpreter.invoke()
            tflite_pred_norm = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # 计算标准化空间的误差
            norm_mae = np.mean(np.abs(pytorch_pred_norm - tflite_pred_norm))
            normalized_mae_errors.append(norm_mae)
            
            # 反标准化预测结果
            pytorch_pred_orig = data_processor.inverse_transform_params(pytorch_pred_norm.reshape(1, -1))[0]
            tflite_pred_orig = data_processor.inverse_transform_params(tflite_pred_norm.reshape(1, -1))[0]
            
            # 计算原始空间的误差
            mae = np.mean(np.abs(pytorch_pred_orig - tflite_pred_orig))
            mae_errors.append(mae)
            
            pytorch_predictions.append(pytorch_pred_orig)
            tflite_predictions.append(tflite_pred_orig)
            
            # 显示对比结果
            print(f"\n--- 测试样本 {i+1}: {description} ---")
            glucose_orig = data_processor.inverse_transform_glucose(glucose_seq.reshape(1, -1))[0]
            print(f"输入血糖: {glucose_orig.round(2)}")
            print(f"PyTorch预测: [频率={pytorch_pred_orig[0]:.2f}Hz, 电流={pytorch_pred_orig[1]:.2f}mA, "
                  f"时长={pytorch_pred_orig[2]:.1f}min, 脉宽={pytorch_pred_orig[3]:.0f}μs, 周期={pytorch_pred_orig[4]:.1f}周]")
            print(f"TFLite预测:  [频率={tflite_pred_orig[0]:.2f}Hz, 电流={tflite_pred_orig[1]:.2f}mA, "
                  f"时长={tflite_pred_orig[2]:.1f}min, 脉宽={tflite_pred_orig[3]:.0f}μs, 周期={tflite_pred_orig[4]:.1f}周]")
            print(f"标准化空间MAE: {norm_mae:.6f}")
            print(f"原始空间MAE: {mae:.4f}")
        
        # 计算总体统计
        overall_mae = np.mean(mae_errors)
        max_mae = np.max(mae_errors)
        min_mae = np.min(mae_errors)
        
        overall_norm_mae = np.mean(normalized_mae_errors)
        max_norm_mae = np.max(normalized_mae_errors)
        min_norm_mae = np.min(normalized_mae_errors)
        
        print(f"\n=== 对比测试总结 ===")
        print(f"测试样本数: {len(test_samples)}")
        print(f"标准化空间误差:")
        print(f"  平均MAE: {overall_norm_mae:.6f}")
        print(f"  最大MAE: {max_norm_mae:.6f}")
        print(f"  最小MAE: {min_norm_mae:.6f}")
        print(f"原始空间误差:")
        print(f"  平均MAE: {overall_mae:.4f}")
        print(f"  最大MAE: {max_mae:.4f}")
        print(f"  最小MAE: {min_mae:.4f}")
        
        # 保存对比结果
        comparison_results = {
            'test_samples_count': len(test_samples),
            'normalized_space': {
                'overall_mae': float(overall_norm_mae),
                'max_mae': float(max_norm_mae),
                'min_mae': float(min_norm_mae),
                'mae_errors': [float(x) for x in normalized_mae_errors]
            },
            'original_space': {
                'overall_mae': float(overall_mae),
                'max_mae': float(max_mae),
                'min_mae': float(min_mae),
                'mae_errors': [float(x) for x in mae_errors]
            },
            'pytorch_predictions': [x.tolist() for x in pytorch_predictions],
            'tflite_predictions': [x.tolist() for x in tflite_predictions]
        }
        
        comparison_path = os.path.join(self.output_dir, f"model_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"对比结果已保存: {comparison_path}")
        
        # 判断转换质量 - 基于标准化空间的误差
        if overall_norm_mae < 1e-4:
            print("✓ 转换质量: 优秀 (标准化MAE < 1e-4)")
            quality_status = "excellent"
        elif overall_norm_mae < 1e-3:
            print("✓ 转换质量: 良好 (标准化MAE < 1e-3)")
            quality_status = "good"
        elif overall_norm_mae < 1e-2:
            print("⚠ 转换质量: 一般 (标准化MAE < 1e-2)")
            quality_status = "fair"
        else:
            print("✗ 转换质量: 较差 (标准化MAE >= 1e-2)")
            quality_status = "poor"
        
        # 保存质量状态到结果中
        comparison_results['conversion_quality'] = quality_status
        
        # 重新保存包含质量状态的结果
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        return comparison_results
    
    def _generate_test_samples(self, data_processor):
        """生成测试样本"""
        test_samples = []
        
        # 测试样本1：正常空腹血糖
        glucose_1 = np.array([5.2, 5.1, 5.3, 5.0, 5.2, 5.4, 5.1, 5.0, 5.3, 5.2, 5.1, 5.0])
        glucose_1_norm = data_processor.glucose_scaler.transform(glucose_1.reshape(1, -1))[0]
        test_samples.append((glucose_1_norm, "正常空腹血糖"))
        
        # 测试样本2：糖尿病高血糖
        glucose_2 = np.array([15.2, 16.1, 15.8, 16.5, 15.9, 16.2, 15.7, 16.0, 15.8, 16.1, 15.9, 16.0])
        glucose_2_norm = data_processor.glucose_scaler.transform(glucose_2.reshape(1, -1))[0]
        test_samples.append((glucose_2_norm, "糖尿病高血糖"))
        
        # 测试样本3：餐后血糖升高
        glucose_3 = np.array([7.0, 8.5, 10.2, 12.1, 11.8, 10.5, 9.2, 8.5, 7.8, 7.2, 6.9, 6.8])
        glucose_3_norm = data_processor.glucose_scaler.transform(glucose_3.reshape(1, -1))[0]
        test_samples.append((glucose_3_norm, "餐后血糖升高"))
        
        # 测试样本4：血糖波动大
        glucose_4 = np.array([8.0, 12.5, 6.8, 14.2, 7.5, 11.8, 9.2, 13.1, 8.5, 10.8, 9.5, 11.2])
        glucose_4_norm = data_processor.glucose_scaler.transform(glucose_4.reshape(1, -1))[0]
        test_samples.append((glucose_4_norm, "血糖波动较大"))
        
        # 测试样本5：低血糖
        glucose_5 = np.array([3.8, 3.5, 3.9, 3.6, 3.7, 3.8, 3.5, 3.6, 3.9, 3.7, 3.8, 3.6])
        glucose_5_norm = data_processor.glucose_scaler.transform(glucose_5.reshape(1, -1))[0]
        test_samples.append((glucose_5_norm, "低血糖状态"))
        
        return test_samples
    
    def run_conversion(self):
        """运行完整的转换流程 - 改进版"""
        print("=== PyTorch到TensorFlow Lite转换开始 (改进版) ===")
        
        try:
            # 1. 加载模型配置
            model_config = self.load_model_config()
            
            # 2. 加载PyTorch模型
            pytorch_model = self.load_pytorch_model(model_config)
            
            # 3. 加载数据处理器
            data_processor = self.load_data_processor()
            
            # 4. 创建TensorFlow模型
            tf_model = self.create_tensorflow_model(pytorch_model, model_config)
            
            # 5. 改进的权重转移
            weight_transfer_success = self.transfer_weights_improved(pytorch_model, tf_model)
            
            if not weight_transfer_success:
                print("⚠ 权重转移失败，转换可能不准确")
                # 重命名文件夹为失败状态
                failed_dir = os.path.join(self.base_output_dir, f"conversion_{self.conversion_timestamp}_FAILED")
                os.rename(self.output_dir, failed_dir)
                print(f"转换失败，文件保存在: {failed_dir}")
                return False
            
            # 6. 转换为TFLite
            tflite_path, model_info = self.convert_to_tflite(tf_model, model_config)
            
            # 7. 模型对比测试
            if tflite_path:
                comparison_results = self.test_model_comparison(
                    pytorch_model, tflite_path, data_processor, model_config
                )
                
                # 根据转换质量重命名文件夹
                quality_status = comparison_results['conversion_quality']
                quality_suffix = {
                    'excellent': 'EXCELLENT',
                    'good': 'GOOD', 
                    'fair': 'FAIR',
                    'poor': 'POOR'
                }
                
                final_dir = os.path.join(self.base_output_dir, 
                                       f"conversion_{self.conversion_timestamp}_{quality_suffix[quality_status]}")
                os.rename(self.output_dir, final_dir)
                
                # 创建转换摘要文件
                self._create_conversion_summary(final_dir, model_config, model_info, comparison_results)
                
                print(f"\n=== 转换完成 ===")
                print(f"TFLite模型: {os.path.join(final_dir, 'tavns_model_improved.tflite')}")
                print(f"模型大小: {model_info['model_size_bytes']:,} 字节")
                print(f"标准化空间预测误差: {comparison_results['normalized_space']['overall_mae']:.6f}")
                print(f"原始空间预测误差: {comparison_results['original_space']['overall_mae']:.4f}")
                print(f"转换质量: {quality_status.upper()}")
                print(f"所有文件保存在: {final_dir}")
                
                return True
            else:
                print("转换失败")
                # 重命名文件夹为失败状态
                failed_dir = os.path.join(self.base_output_dir, f"conversion_{self.conversion_timestamp}_FAILED")
                os.rename(self.output_dir, failed_dir)
                print(f"转换失败，文件保存在: {failed_dir}")
                return False
                
        except Exception as e:
            print(f"转换过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            # 重命名文件夹为错误状态
            error_dir = os.path.join(self.base_output_dir, f"conversion_{self.conversion_timestamp}_ERROR")
            if os.path.exists(self.output_dir):
                os.rename(self.output_dir, error_dir)
                print(f"转换出错，文件保存在: {error_dir}")
            return False
    
    def _create_conversion_summary(self, output_dir, model_config, model_info, comparison_results):
        """创建转换摘要文件"""
        summary = {
            'conversion_info': {
                'timestamp': self.conversion_timestamp,
                'source_model': self.model_dir,
                'conversion_quality': comparison_results['conversion_quality'],
                'success': True
            },
            'model_details': {
                'input_dim': model_config['input_dim'],
                'param_dim': model_config['param_dim'],
                'hidden_dim': model_config['hidden_dim'],
                'model_size_bytes': model_info['model_size_bytes'],
                'model_size_kb': round(model_info['model_size_bytes'] / 1024, 1)
            },
            'accuracy_metrics': {
                'normalized_space_mae': comparison_results['normalized_space']['overall_mae'],
                'original_space_mae': comparison_results['original_space']['overall_mae'],
                'test_samples_count': comparison_results['test_samples_count']
            },
            'output_files': {
                'tflite_model': 'tavns_model_improved.tflite',
                'model_info': 'model_info.json',
                'comparison_results': 'model_comparison.json',
                'summary': 'conversion_summary.json'
            }
        }
        
        summary_path = os.path.join(output_dir, 'conversion_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"转换摘要已保存: {summary_path}")

def main():
    """主函数"""
    print("=== taVNS模型TensorFlow Lite转换工具 (改进版) ===")
    
    # 创建转换器
    converter = PyTorchToTFLiteConverter()
    
    # 运行转换
    success = converter.run_conversion()
    
    if success:
        print("\n✓ 转换成功完成！")
        print(f"输出文件位于: {converter.output_dir}")
    else:
        print("\n✗ 转换失败")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 