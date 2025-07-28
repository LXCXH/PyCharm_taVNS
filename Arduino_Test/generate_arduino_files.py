#!/usr/bin/env python3
"""
Arduino文件生成器
自动生成ESP32-S3 Arduino项目所需的模型数据和标准化参数文件

使用方法:
python generate_arduino_files.py

要求:
1. 已运行convert_to_tflite.py生成TFLite模型
2. Training_Outputs目录中有最新的训练结果
"""

import os
import pickle
import json
import glob
import sys
from datetime import datetime

# 添加项目根目录到Python路径，以便导入data_processor模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

def find_latest_training_output():
    """查找最新的训练输出目录"""
    training_dirs = glob.glob('Training_Outputs/training_output_*')
    if not training_dirs:
        print("❌ 未找到训练输出目录")
        return None
    
    # 按时间戳排序，获取最新的
    latest_dir = max(training_dirs, key=lambda x: os.path.getctime(x))
    print(f"✅ 找到最新训练目录: {latest_dir}")
    return latest_dir

def find_latest_tflite_model(prefer_float32=False):
    """查找最新的TFLite模型文件
    
    Args:
        prefer_float32: 是否优先选择Float32非量化模型
    """
    tflite_dirs = glob.glob('TFLite_Output/conversion_*')
    if not tflite_dirs:
        print("❌ 未找到TFLite转换目录")
        print("请先运行 python convert_to_tflite.py")
        return None
    
    # 按时间戳排序，获取最新的
    latest_dir = max(tflite_dirs, key=lambda x: os.path.getctime(x))
    
    # 根据偏好选择模型文件
    model_candidates = []
    
    if prefer_float32:
        # 优先尝试Float32模型
        float32_path = os.path.join(latest_dir, 'tavns_model_float32.tflite')
        if os.path.exists(float32_path):
            print(f"✅ 找到Float32非量化模型: {float32_path}")
            return float32_path
    
    # 尝试improved模型
    improved_path = os.path.join(latest_dir, 'tavns_model_improved.tflite')
    if os.path.exists(improved_path):
        print(f"✅ 找到量化优化模型: {improved_path}")
        return improved_path
    
    # 查找其他.tflite文件
    tflite_files = glob.glob(os.path.join(latest_dir, '*.tflite'))
    if tflite_files:
        tflite_path = tflite_files[0]
        print(f"✅ 找到TFLite模型: {tflite_path}")
        return tflite_path
    
    print(f"❌ 在{latest_dir}中未找到TFLite模型文件")
    return None

def generate_model_data_header(tflite_path, output_path):
    """生成模型数据头文件"""
    print(f"📄 生成模型数据头文件: {output_path}")
    
    try:
        with open(tflite_path, 'rb') as f:
            data = f.read()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('#ifndef TAVNS_MODEL_DATA_H\n')
            f.write('#define TAVNS_MODEL_DATA_H\n\n')
            
            # 检测模型类型
            model_type = "Float32非量化" if 'float32' in os.path.basename(tflite_path) else "量化优化"
            esp32_optimized = "是" if 'float32' in os.path.basename(tflite_path) else "否"
            
            f.write('/*\n')
            f.write(' * TensorFlow Lite模型数据\n')
            f.write(f' * 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f' * 源文件: {tflite_path}\n')
            f.write(f' * 模型大小: {len(data):,} 字节\n')
            f.write(f' * 模型类型: {model_type}\n')
            f.write(f' * ESP32-S3优化: {esp32_optimized}\n')
            f.write(' * \n')
            f.write(' * 此文件由generate_arduino_files.py自动生成\n')
            f.write(' */\n\n')
            
            f.write('const unsigned char tavns_model_data[] = {\n')
            
            for i, byte in enumerate(data):
                if i % 12 == 0:
                    f.write('  ')
                f.write(f'0x{byte:02x}')
                if i < len(data) - 1:
                    f.write(', ')
                if (i + 1) % 12 == 0:
                    f.write('\n')
            
            if len(data) % 12 != 0:
                f.write('\n')
            
            f.write('};\n\n')
            f.write(f'const unsigned int tavns_model_data_len = {len(data)};\n\n')
            
            # 添加模型信息
            f.write('// 模型信息\n')
            f.write('#define MODEL_INPUT_SIZE 12    // 血糖序列长度\n')
            f.write('#define MODEL_OUTPUT_SIZE 5    // 输出参数数量\n')
            f.write('#define MODEL_VERSION 3        // TensorFlow Lite Schema版本\n')
            f.write(f'#define MODEL_SIZE_BYTES {len(data)}  // 模型大小（字节）\n\n')
            
            f.write('#endif // TAVNS_MODEL_DATA_H\n')
        
        print(f"✅ 模型数据头文件生成完成 ({len(data):,} 字节)")
        return True
        
    except Exception as e:
        print(f"❌ 生成模型数据头文件失败: {e}")
        return False

def generate_scaler_params_from_config(config_path, output_path):
    """从训练配置文件生成标准化参数头文件（备用方案）"""
    print(f"📄 从配置文件生成标准化参数: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 使用默认的标准化参数（基于论文数据）
        glucose_mean = [10.5] * 12  # 默认血糖均值
        glucose_std = [5.2] * 12    # 默认血糖标准差
        param_min = [1.0, 0.5, 10.0, 50.0, 1.0]      # [频率, 电流, 时长, 脉宽, 周期]
        param_max = [50.0, 5.0, 60.0, 2000.0, 20.0]  # [频率, 电流, 时长, 脉宽, 周期]
        
        print("⚠️  使用默认标准化参数（基于论文数据）")
        print("   如需精确参数，请确保data_processor.pkl文件可用")
        
        return write_scaler_params_header(output_path, glucose_mean, glucose_std, param_min, param_max, 
                                        source_info=f"默认参数 (来源: {config_path})")
        
    except Exception as e:
        print(f"❌ 从配置文件生成标准化参数失败: {e}")
        return False

def write_scaler_params_header(output_path, glucose_mean, glucose_std, param_min, param_max, source_info=""):
    """写入标准化参数头文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('#ifndef SCALER_PARAMS_H\n')
            f.write('#define SCALER_PARAMS_H\n\n')
            
            f.write('/*\n')
            f.write(' * 数据标准化参数\n')
            f.write(f' * 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f' * 参数来源: {source_info}\n')
            f.write(' * \n')
            f.write(' * 此文件由generate_arduino_files.py自动生成\n')
            f.write(' * 请勿手动修改，重新生成时会覆盖\n')
            f.write(' */\n\n')
            
            # 血糖标准化参数
            f.write('// =============================================================================\n')
            f.write('// 血糖数据标准化参数 (StandardScaler)\n')
            f.write('// =============================================================================\n\n')
            
            f.write('// 血糖序列的均值 (用于标准化: (x - mean) / std)\n')
            f.write('const float glucose_scaler_mean[12] = {\n')
            mean_values = [f'{val:.6f}' for val in glucose_mean]
            f.write('  ' + ', '.join(mean_values) + '\n')
            f.write('};\n\n')
            
            f.write('// 血糖序列的标准差 (用于标准化: (x - mean) / std)\n')
            f.write('const float glucose_scaler_std[12] = {\n')
            std_values = [f'{val:.6f}' for val in glucose_std]
            f.write('  ' + ', '.join(std_values) + '\n')
            f.write('};\n\n')
            
            # 参数标准化参数
            f.write('// =============================================================================\n')
            f.write('// 参数数据标准化参数 (MinMaxScaler)\n')
            f.write('// =============================================================================\n\n')
            
            f.write('// 参数的最小值 (用于反标准化: x * (max - min) + min)\n')
            f.write('const float param_scaler_min[5] = {\n')
            f.write('  // 顺序: [频率, 电流, 时长, 脉宽, 周期]\n')
            min_values = [f'{val:.6f}' for val in param_min]
            f.write('  ' + ', '.join(min_values) + '\n')
            f.write('};\n\n')
            
            f.write('// 参数的最大值 (用于反标准化: x * (max - min) + min)\n')
            f.write('const float param_scaler_max[5] = {\n')
            f.write('  // 顺序: [频率, 电流, 时长, 脉宽, 周期]\n')
            max_values = [f'{val:.6f}' for val in param_max]
            f.write('  ' + ', '.join(max_values) + '\n')
            f.write('};\n\n')
            
            # 参数名称和单位
            f.write('// =============================================================================\n')
            f.write('// 参数名称和单位 (用于显示)\n')
            f.write('// =============================================================================\n\n')
            
            f.write('const char* param_names[5] = {\n')
            f.write('  "频率", "电流", "时长", "脉宽", "周期"\n')
            f.write('};\n\n')
            
            f.write('const char* param_units[5] = {\n')
            f.write('  "Hz", "mA", "min", "μs", "周"\n')
            f.write('};\n\n')
            
            # 辅助函数
            f.write('// =============================================================================\n')
            f.write('// 辅助函数\n')
            f.write('// =============================================================================\n\n')
            
            f.write('// 血糖数据标准化函数\n')
            f.write('inline void normalize_glucose_value(float raw_value, int index, float* normalized_value) {\n')
            f.write('  *normalized_value = (raw_value - glucose_scaler_mean[index]) / glucose_scaler_std[index];\n')
            f.write('}\n\n')
            
            f.write('// 参数数据反标准化函数\n')
            f.write('inline void denormalize_param_value(float normalized_value, int index, float* param_value) {\n')
            f.write('  *param_value = normalized_value * (param_scaler_max[index] - param_scaler_min[index]) + param_scaler_min[index];\n')
            f.write('}\n\n')
            
            f.write('// 验证参数范围是否合理\n')
            f.write('inline bool validate_param_range(float param_value, int index) {\n')
            f.write('  return (param_value >= param_scaler_min[index] && param_value <= param_scaler_max[index]);\n')
            f.write('}\n\n')
            
            f.write('#endif // SCALER_PARAMS_H\n')
        
        return True
        
    except Exception as e:
        print(f"❌ 写入标准化参数头文件失败: {e}")
        return False

def generate_scaler_params_header(training_dir, output_path):
    """生成标准化参数头文件"""
    print(f"📄 生成标准化参数头文件: {output_path}")
    
    data_processor_path = os.path.join(training_dir, 'data_processor.pkl')
    
    if not os.path.exists(data_processor_path):
        print(f"❌ 数据处理器文件不存在: {data_processor_path}")
        # 尝试查找training_config.json文件中的标准化参数
        config_path = os.path.join(training_dir, 'training_config.json')
        if os.path.exists(config_path):
            print("⚠️  尝试从training_config.json获取标准化参数")
            return generate_scaler_params_from_config(config_path, output_path)
        return False
    
    try:
        # 尝试导入data_processor模块以支持pickle反序列化
        try:
            import data_processor
            print("✅ 成功导入data_processor模块")
        except ImportError:
            print("⚠️  无法导入data_processor模块，尝试使用兼容模式")
        
        # 使用weights_only=False来避免pickle安全限制
        with open(data_processor_path, 'rb') as f:
            try:
                processor = pickle.load(f)
                print("✅ 成功加载data_processor.pkl文件")
            except Exception as e:
                print(f"❌ 加载pickle文件失败: {e}")
                # 尝试使用torch.load加载（如果是PyTorch保存的）
                try:
                    import torch
                    processor = torch.load(data_processor_path, map_location='cpu', weights_only=False)
                    print("✅ 使用torch.load成功加载文件")
                except Exception as torch_e:
                    print(f"❌ torch.load也失败: {torch_e}")
                    raise e
        
        # 提取标准化参数
        try:
            glucose_mean = processor.glucose_scaler.mean_
            glucose_std = processor.glucose_scaler.scale_
            param_min = processor.param_scaler.data_min_
            param_max = processor.param_scaler.data_max_
            print("✅ 成功提取标准化参数")
        except AttributeError as attr_e:
            print(f"❌ 提取标准化参数失败: {attr_e}")
            print("尝试打印processor的属性...")
            print(f"processor类型: {type(processor)}")
            if hasattr(processor, '__dict__'):
                print(f"processor属性: {list(processor.__dict__.keys())}")
            raise attr_e
        
        # 使用共用的写入函数
        success = write_scaler_params_header(output_path, glucose_mean, glucose_std, param_min, param_max, 
                                           source_info=f"训练数据 (来源: {training_dir})")
        
        if success:
            # 打印参数摘要
            print("✅ 标准化参数头文件生成完成")
            print("📊 参数摘要:")
            print(f"   血糖均值范围: {glucose_mean.min():.2f} ~ {glucose_mean.max():.2f}")
            print(f"   血糖标准差范围: {glucose_std.min():.2f} ~ {glucose_std.max():.2f}")
            print(f"   参数最小值: [{param_min[0]:.1f}, {param_min[1]:.1f}, {param_min[2]:.1f}, {param_min[3]:.0f}, {param_min[4]:.1f}]")
            print(f"   参数最大值: [{param_max[0]:.1f}, {param_max[1]:.1f}, {param_max[2]:.1f}, {param_max[3]:.0f}, {param_max[4]:.1f}]")
        
        return success
        
    except Exception as e:
        print(f"❌ 生成标准化参数头文件失败: {e}")
        return False

def generate_conversion_info(training_dir, tflite_path, output_path):
    """生成转换信息文件"""
    print(f"📄 生成转换信息文件: {output_path}")
    
    try:
        # 获取模型大小
        model_size = os.path.getsize(tflite_path)
        
        # 尝试读取训练配置
        config_path = os.path.join(training_dir, 'training_config.json')
        training_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                training_config = json.load(f)
        
        # 尝试读取评估结果
        eval_path = os.path.join(training_dir, 'evaluation_results.json')
        eval_results = {}
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
        
        conversion_info = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'script_version': '1.0',
                'training_directory': training_dir,
                'tflite_model_path': tflite_path
            },
            'model_info': {
                'input_size': 12,
                'output_size': 5,
                'model_size_bytes': model_size,
                'model_size_kb': round(model_size / 1024, 1)
            },
            'training_config': training_config,
            'evaluation_results': eval_results,
            'arduino_files': {
                'main_program': 'esp32_tavns_test.ino',
                'model_data': 'tavns_model_data.h',
                'scaler_params': 'scaler_params.h',
                'readme': 'README.md'
            },
            'usage_instructions': {
                'step1': '在Arduino IDE中打开esp32_tavns_test.ino',
                'step2': '选择ESP32S3 Dev Module开发板',
                'step3': '配置PSRAM和分区方案',
                'step4': '编译并上传到ESP32-S3',
                'step5': '打开串口监视器查看测试结果'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversion_info, f, indent=2, ensure_ascii=False)
        
        print("✅ 转换信息文件生成完成")
        return True
        
    except Exception as e:
        print(f"❌ 生成转换信息文件失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Arduino文件生成器')
    parser.add_argument('--float32', action='store_true', 
                       help='优先使用Float32非量化模型（默认，专为ESP32-S3优化）')
    parser.add_argument('--quantized', action='store_true', 
                       help='优先使用量化优化模型')
    
    args = parser.parse_args()
    
    # 确定模型偏好 - 默认优先Float32
    if args.quantized:
        prefer_float32 = False
        mode_desc = "量化优化"
    else:
        prefer_float32 = True
        mode_desc = "Float32非量化（默认）"
    
    print("🚀 Arduino文件生成器")
    print("=" * 50)
    print(f"🔧 模型偏好: {mode_desc}模式")
    
    # 检查Arduino_Test目录
    arduino_dir = 'Arduino_Test'
    if not os.path.exists(arduino_dir):
        print(f"❌ Arduino_Test目录不存在")
        return False
    
    # 查找最新的训练输出
    training_dir = find_latest_training_output()
    if not training_dir:
        return False
    
    # 查找最新的TFLite模型
    tflite_path = find_latest_tflite_model(prefer_float32)
    if not tflite_path:
        return False
    
    print("\n📁 开始生成Arduino文件...")
    
    success_count = 0
    total_count = 3
    
    # 生成模型数据头文件
    if generate_model_data_header(tflite_path, os.path.join(arduino_dir, 'tavns_model_data.h')):
        success_count += 1
    
    # 生成标准化参数头文件
    if generate_scaler_params_header(training_dir, os.path.join(arduino_dir, 'scaler_params.h')):
        success_count += 1
    
    # 生成转换信息文件
    if generate_conversion_info(training_dir, tflite_path, os.path.join(arduino_dir, 'conversion_info.json')):
        success_count += 1
    
    print("\n" + "=" * 50)
    if success_count == total_count:
        # 检测使用的模型类型
        model_type = "Float32非量化" if 'float32' in os.path.basename(tflite_path) else "量化优化"
        model_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        
        print("🎉 所有Arduino文件生成完成!")
        print(f"📂 输出目录: {arduino_dir}")
        print(f"🔧 模型类型: {model_type} ({model_size_mb:.1f} MB)")
        
        print("\n📋 生成的文件:")
        print(f"   ✅ tavns_model_data.h - TensorFlow Lite模型数据 ({model_type})")
        print("   ✅ scaler_params.h - 数据标准化参数") 
        print("   ✅ conversion_info.json - 转换信息")
        
        print("\n🔧 下一步:")
        print("   1. 在Arduino IDE中打开esp32_tavns_test.ino")
        print("   2. 选择ESP32S3开发板并配置参数")
        print("   3. 编译并上传到ESP32-S3")
        print("   4. 打开串口监视器查看测试结果")
        
        if 'float32' in os.path.basename(tflite_path):
            print("\n💡 提示: 使用Float32非量化模型（默认），专为ESP32-S3 TFLite Micro优化")
        else:
            print("\n💡 提示: 如果ESP32输出固定值，建议使用默认的Float32非量化模式")
            
        return True
    else:
        print(f"⚠️  部分文件生成失败 ({success_count}/{total_count})")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 