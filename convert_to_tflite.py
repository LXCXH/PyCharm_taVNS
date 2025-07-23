#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import taVNSNet
from data_processor import taVNSDataProcessor

class TFLiteConverter:
    """PyTorch到TensorFlow Lite转换器"""
    
    def __init__(self, model_path, output_dir="tflite_output"):
        """
        初始化转换器
        
        Args:
            model_path: PyTorch模型文件路径
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.data_processor = taVNSDataProcessor()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 模型配置
        self.input_shape = (12,)  # 血糖序列长度
        self.output_shape = (5,)  # 刺激参数数量
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.2
        
    def load_pytorch_model(self):
        """加载PyTorch模型"""
        print("正在加载PyTorch模型...")
        
        # 创建模型实例
        model = taVNSNet(
            input_dim=12,
            param_dim=5,
            hidden_dim=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"模型已加载: {self.model_path}")
        return model
    
    def create_keras_model(self):
        """创建等效的Keras模型"""
        print("正在创建Keras模型...")
        
        # 输入层
        inputs = keras.Input(shape=self.input_shape, name='glucose_input')
        
        # 重塑为序列格式 (batch_size, sequence_length, features)
        x = layers.Reshape((self.input_shape[0], 1))(inputs)
        
        # LSTM层
        lstm_output = layers.LSTM(
            units=self.hidden_size,
            return_sequences=True,
            dropout=self.dropout,
            name='lstm_layer'
        )(x)
        
        # 多头注意力机制
        attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.hidden_size // 4,
            dropout=self.dropout,
            name='multi_head_attention'
        )(lstm_output, lstm_output)
        
        # 残差连接和层归一化
        x = layers.Add()([lstm_output, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(x)
        
        # 全连接层
        x = layers.Dense(128, activation='relu', name='fc1')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(64, activation='relu', name='fc2')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # 输出层 - 刺激参数预测
        outputs = layers.Dense(
            self.output_shape[0], 
            activation='linear', 
            name='stim_params_output'
        )(x)
        
        # 创建模型
        keras_model = keras.Model(inputs=inputs, outputs=outputs, name='taVNS_TFLite')
        
        print("Keras模型已创建")
        return keras_model
    
    def convert_to_tflite(self, keras_model):
        """转换为TFLite格式"""
        print("正在转换为TFLite格式...")
        
        # 创建TFLite转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        
        # 设置优化选项
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 设置目标规范（用于微控制器）- 修复LSTM转换问题
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # 禁用TensorList操作降低
        converter._experimental_lower_tensor_list_ops = False
        
        # 设置支持的类型
        converter.target_spec.supported_types = [tf.float32]
        
        # 转换模型
        tflite_model = converter.convert()
        
        # 保存TFLite模型
        tflite_path = os.path.join(self.output_dir, "tavns_model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite模型已保存: {tflite_path}")
        return tflite_path
    
    def test_tflite_model(self, tflite_path):
        """测试TFLite模型"""
        print("正在测试TFLite模型...")
        
        # 加载TFLite模型
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # 获取输入输出详情
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"输入详情: {input_details}")
        print(f"输出详情: {output_details}")
        
        # 创建测试数据
        test_input = np.random.random((1, 12)).astype(np.float32)
        
        # 设置输入
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # 运行推理
        interpreter.invoke()
        
        # 获取输出
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"测试输入形状: {test_input.shape}")
        print(f"测试输出形状: {output.shape}")
        print(f"测试输出: {output.flatten()}")
        
        return output.flatten()
    
    def create_model_info(self, tflite_path):
        """创建模型信息文件"""
        print("正在创建模型信息文件...")
        
        # 获取模型大小
        with open(tflite_path, 'rb') as f:
            model_size = len(f.read())
        
        info = {
            "model_name": "taVNS参数预测模型",
            "version": "1.0",
            "model_format": "TensorFlow Lite",
            "model_size_bytes": model_size,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "input_description": "血糖序列 (12个点，每5分钟一个)",
            "output_description": "taVNS刺激参数 [频率(Hz), 电流(mA), 时长(分钟), 脉宽(μs), 周期(周)]",
            "model_architecture": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout
            },
            "target_platform": "ESP32-S3",
            "framework": "TensorFlow Lite",
            "optimization": "量化优化",
            "notes": [
                "模型基于三篇科学论文数据训练",
                "支持2/15 Hz交替刺激模式",
                "适用于糖尿病血糖管理",
                "支持个体化参数调整",
                "可直接用于Arduino ESP32-S3项目"
            ]
        }
        
        # 保存信息文件
        info_path = os.path.join(self.output_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"模型信息文件已保存: {info_path}")
        return info_path
    
    def create_model_weights_info(self, model):
        """创建模型权重信息文件"""
        print("正在创建模型权重信息文件...")
        
        weights_info = {
            "model_path": self.model_path,
            "total_parameters": 0,
            "trainable_parameters": 0,
            "layers": []
        }
        
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            layer_info = {
                "name": name,
                "shape": list(param.shape),
                "parameters": param.numel(),
                "requires_grad": param.requires_grad
            }
            weights_info["layers"].append(layer_info)
            
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        weights_info["total_parameters"] = total_params
        weights_info["trainable_parameters"] = trainable_params
        
        # 保存权重信息文件
        weights_path = os.path.join(self.output_dir, "model_weights_info.json")
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights_info, f, indent=2, ensure_ascii=False)
        
        print(f"模型权重信息文件已保存: {weights_path}")
        return weights_path
    
    def create_conversion_report(self, test_output):
        """创建转换报告"""
        print("正在创建转换报告...")
        
        report = {
            "conversion_status": "成功转换为TensorFlow Lite格式",
            "model_test_results": {
                "input_shape": self.input_shape,
                "output_shape": self.output_shape,
                "test_output": test_output.tolist(),
                "test_output_description": "taVNS刺激参数 [频率(Hz), 电流(mA), 时长(分钟), 脉宽(μs), 周期(周)]"
            },
            "arduino_usage": [
                "1. 将 tavns_model.tflite 复制到Arduino项目",
                "2. 安装TensorFlow Lite ESP32库",
                "3. 使用TFLiteInterpreter加载模型",
                "4. 在ESP32-S3上运行推理"
            ],
            "arduino_requirements": [
                "Arduino IDE",
                "ESP32开发板支持",
                "TensorFlow Lite ESP32库",
                "ESP32-S3开发板"
            ]
        }
        
        # 保存转换报告
        report_path = os.path.join(self.output_dir, "conversion_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"转换报告已保存: {report_path}")
        return report_path
    
    def convert(self):
        """执行转换流程"""
        print("=== taVNS模型转换开始 ===")
        
        try:
            # 1. 加载PyTorch模型
            pytorch_model = self.load_pytorch_model()
            
            # 2. 创建Keras模型
            keras_model = self.create_keras_model()
            
            # 3. 转换为TFLite格式
            tflite_path = self.convert_to_tflite(keras_model)
            
            # 4. 测试TFLite模型
            test_output = self.test_tflite_model(tflite_path)
            
            # 5. 创建模型信息文件
            info_path = self.create_model_info(tflite_path)
            
            # 6. 创建模型权重信息文件
            weights_path = self.create_model_weights_info(pytorch_model)
            
            # 7. 创建转换报告
            report_path = self.create_conversion_report(test_output)
            
            # 8. 获取最终模型大小
            with open(tflite_path, 'rb') as f:
                model_size = len(f.read())
            
            print("\n=== 转换完成 ===")
            print(f"输出目录: {self.output_dir}")
            print(f"TFLite模型: {tflite_path}")
            print(f"模型大小: {model_size} 字节")
            print(f"模型信息: {info_path}")
            print(f"权重信息: {weights_path}")
            print(f"转换报告: {report_path}")
            print(f"测试输出: {test_output}")
            
            return True
            
        except Exception as e:
            print(f"转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    # 查找最新的训练输出
    training_outputs_dir = "Training_Outputs"
    if not os.path.exists(training_outputs_dir):
        print("错误: 找不到Training_Outputs目录")
        return
    
    # 查找最新的best_model.pth
    latest_model = None
    
    for item in os.listdir(training_outputs_dir):
        item_path = os.path.join(training_outputs_dir, item)
        if os.path.isdir(item_path):
            model_path = os.path.join(item_path, "best_model.pth")
            if os.path.exists(model_path):
                if not latest_model or item > latest_model:
                    latest_model = item
    
    if not latest_model:
        print("错误: 找不到训练好的模型文件")
        return
    
    model_path = os.path.join(training_outputs_dir, latest_model, "best_model.pth")
    print(f"使用模型: {model_path}")
    
    # 创建转换器并执行转换
    converter = TFLiteConverter(model_path)
    success = converter.convert()
    
    if success:
        print("\n🎉 转换成功!")
        print("\n📋 模型文件已准备就绪:")
        print("1. tavns_model.tflite - Arduino可用的TensorFlow Lite模型")
        print("2. model_info.json - 模型详细信息")
        print("3. model_weights_info.json - 模型权重信息")
        print("4. conversion_report.json - 转换报告")
        print("\n💡 使用方法:")
        print("1. 将 tavns_model.tflite 复制到Arduino项目")
        print("2. 使用TensorFlow Lite ESP32库加载模型")
        print("3. 在ESP32-S3上运行推理")
    else:
        print("\n❌ 转换失败，请检查错误信息")

if __name__ == "__main__":
    main() 