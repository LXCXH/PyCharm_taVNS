import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class taVNSDataProcessor:
    """
    taVNS数据处理器
    基于三篇论文的实际实验数据重新构建
    专门用于taVNS参数预测
    """
    
    def __init__(self, target_sequence_length=12):
        self.target_sequence_length = target_sequence_length
        self.glucose_scaler = StandardScaler()
        self.param_scaler = MinMaxScaler()
        
        # 基于论文1：ZDF小鼠实验数据
        self.paper1_data = {
            'study_type': 'ZDF_mice',
            'stimulation_params': {
                'frequency_low': 2,    # 2 Hz
                'frequency_high': 15,  # 15 Hz
                'frequency_mode': 'alternating',  # 交替模式：每秒切换
                'amplitude': 2.0,      # 2 mA
                'duration': 30,        # 30分钟
                'pulse_width': 200,    # 假设值
                'session_duration': 5  # 持续5周
            },
            'baseline_glucose': {
                'control': [19, 22, 24, 27, 29, 32],  # 未进行taVNS (Week 0-5)
                'treatment': [19, 10, 11, 12, 12, 11] # 进行taVNS (Week 0-5)
            },
            'pancreatic_removal_glucose': {
                'day1': [25, 24.5, 26.5, 29.5, 27.5, 26, 24.5],
                'day3': [22, 24, 29, 27, 24, 23.5, 23.5],
                'day5': [21, 24, 30, 24.5, 22, 21, 19.5]
            }
        }
        
        # 基于论文2：人体IGT患者实验数据
        self.paper2_data = {
            'study_type': 'human_IGT',
            'stimulation_params': {
                'frequency': 20,    # 20 Hz
                'amplitude': 1.0,   # 1 mA
                'duration': 20,     # 20分钟
                'pulse_width': 1000, # 1 ms = 1000 μs
                'session_duration': 12  # 12周
            },
            'baseline_glucose': {
                'inclusion_criteria': (7.0, 11.1),  # 空腹血糖7.0-11.1 mmol/L
                'meal_response': (7.8, 11.1)        # 餐后2小时血糖7.8-11.1 mmol/L
            },
            '2hPG_results': {
                'baseline': {'ta_vns': 9.7, 'sham': 9.1, 'no_treatment': 9.3},
                '6_weeks': {'ta_vns': 7.3, 'sham': 8.0, 'no_treatment': 9.5},
                '12_weeks': {'ta_vns': 7.5, 'sham': 8.0, 'no_treatment': 10.0}
            }
        }
        
        # 基于论文3：健康人餐后血糖抑制实验
        self.paper3_data = {
            'study_type': 'healthy_postprandial',
            'stimulation_params': {
                'protocol1': {
                    'frequency': 10,     # 10 Hz
                    'amplitude': 2.3,    # 2.0-2.3 mA
                    'duration': 30,      # 30分钟
                    'pulse_width': 300,  # 0.3 ms = 300 μs
                    'timing': 'post_meal'
                },
                'protocol2': {
                    'frequency': 10,     # 10 Hz  
                    'amplitude': 2.3,    # 2.0-2.3 mA
                    'duration': 30,      # 30分钟
                    'pulse_width': 300,  # 0.3 ms = 300 μs
                    'timing': 'pre_meal_220kcal'
                }
            },
            'glucose_response': {
                'protocol1': {
                    'sham': [105, 110, 115, 120, 115, 110, 105, 100, 95, 90, 85, 80],
                    'tavns': [100, 105, 110, 115, 110, 105, 100, 95, 90, 85, 80, 75]
                },
                'protocol2': {
                    'sham': [100, 110, 130, 150, 170, 180, 175, 170, 165, 160, 155, 150],
                    'tavns': [100, 115, 140, 160, 175, 185, 180, 175, 170, 165, 160, 155]
                }
            }
        }
        
    def create_comprehensive_dataset_from_papers(self):
        """
        基于三篇论文的实际数据创建综合训练数据集
        专门用于taVNS参数预测
        """
        samples = []
        
        # 1. 基于论文1的ZDF小鼠数据
        samples.extend(self._create_zdf_mice_samples())
        
        # 2. 基于论文2的人体IGT患者数据
        samples.extend(self._create_human_igt_samples())
        
        # 3. 基于论文3的健康人餐后血糖数据
        samples.extend(self._create_healthy_postprandial_samples())
        
        return samples
    
    def _create_zdf_mice_samples(self):
        """
        基于论文1创建ZDF小鼠实验样本
        """
        samples = []
        data = self.paper1_data
        
        # 长期效应数据（5周实验）
        control_glucose = data['baseline_glucose']['control']
        treatment_glucose = data['baseline_glucose']['treatment']
        
        for week in range(len(control_glucose)):
            # 创建基于周数据的12点时间序列
            base_control = control_glucose[week]
            base_treatment = treatment_glucose[week]
            
            # 模拟1小时内的血糖变化（12个点，每5分钟）
            control_sequence = self._generate_hourly_sequence(base_control, 'high_variation')
            
            # 添加个体差异
            for sensitivity in [0.8, 1.0, 1.2]:
                # 对于交替频率模式，使用专门的调整方法
                avg_frequency, freq_low, freq_high = self._create_alternating_frequency_params(
                    data['stimulation_params']['frequency_low'],
                    data['stimulation_params']['frequency_high'],
                    control_sequence, sensitivity
                )
                
                # 构建刺激参数
                stim_params = np.array([
                    avg_frequency,  # 使用调整后的平均频率
                    data['stimulation_params']['amplitude'],
                    data['stimulation_params']['duration'],
                    data['stimulation_params']['pulse_width'],
                    data['stimulation_params']['session_duration']
                ])
                
                # 进一步调整其他参数
                adjusted_params = self._adjust_params_for_glucose_and_sensitivity(
                    stim_params, control_sequence, sensitivity
                )
                
                samples.append({
                    'input_glucose': control_sequence,
                    'stim_params': adjusted_params,
                    'individual_sensitivity': sensitivity,
                    'study_type': 'ZDF_mice',
                    'week': week,
                    'stimulation_intensity': self._calculate_stimulation_intensity(adjusted_params)
                })
        
        # 胰腺切除实验数据
        pancreatic_data = data['pancreatic_removal_glucose']
        for day, glucose_values in pancreatic_data.items():
            # 扩展到12个点
            extended_sequence = self._extend_to_12_points(glucose_values)
            
            # 对于交替频率模式，使用专门的调整方法
            avg_frequency, freq_low, freq_high = self._create_alternating_frequency_params(
                data['stimulation_params']['frequency_low'],
                data['stimulation_params']['frequency_high'],
                extended_sequence, 0.7
            )
            
            # 构建刺激参数
            stim_params = np.array([
                avg_frequency,  # 使用调整后的平均频率
                data['stimulation_params']['amplitude'],
                data['stimulation_params']['duration'],
                data['stimulation_params']['pulse_width'],
                data['stimulation_params']['session_duration']
            ])
            
            # 进一步调整其他参数
            adjusted_params = self._adjust_params_for_glucose_and_sensitivity(
                stim_params, extended_sequence, 0.7
            )
            
            samples.append({
                'input_glucose': extended_sequence,
                'stim_params': adjusted_params,
                'individual_sensitivity': 0.7,  # 手术后敏感性降低
                'study_type': 'ZDF_mice_pancreatic',
                'day': day,
                'stimulation_intensity': self._calculate_stimulation_intensity(adjusted_params)
            })
        
        return samples
    
    def _create_human_igt_samples(self):
        """
        基于论文2创建人体IGT患者样本
        """
        samples = []
        data = self.paper2_data
        
        # 2hPG实验数据
        results = data['2hPG_results']
        
        # 刺激参数
        stim_params = np.array([
            data['stimulation_params']['frequency'],
            data['stimulation_params']['amplitude'],
            data['stimulation_params']['duration'],
            data['stimulation_params']['pulse_width'],
            data['stimulation_params']['session_duration']
        ])
        
        # 基于不同时间点的数据
        time_points = ['baseline', '6_weeks', '12_weeks']
        
        for time_point in time_points:
            sham_glucose = results[time_point]['sham']
            
            # 生成餐后血糖曲线
            sham_sequence = self._generate_postprandial_curve(sham_glucose, 'IGT_pattern')
            
            # 添加个体差异
            for sensitivity in [0.6, 0.8, 1.0, 1.2, 1.4]:
                # 根据血糖水平和个体敏感性调整刺激参数
                adjusted_params = self._adjust_params_for_glucose_and_sensitivity(
                    stim_params, sham_sequence, sensitivity
                )
                
                samples.append({
                    'input_glucose': sham_sequence,
                    'stim_params': adjusted_params,
                    'individual_sensitivity': sensitivity,
                    'study_type': 'human_IGT',
                    'time_point': time_point,
                    'stimulation_intensity': self._calculate_stimulation_intensity(adjusted_params)
                })
        
        return samples
    
    def _create_healthy_postprandial_samples(self):
        """
        基于论文3创建健康人餐后血糖样本
        """
        samples = []
        data = self.paper3_data
        
        # 两种协议的数据
        for protocol_name, protocol_data in data['stimulation_params'].items():
            # 刺激参数
            stim_params = np.array([
                protocol_data['frequency'],
                protocol_data['amplitude'],
                protocol_data['duration'],
                protocol_data['pulse_width'],
                4  # 假设短期实验周期
            ])
            
            # 血糖响应数据
            glucose_data = data['glucose_response'][protocol_name]
            
            # 转换单位 mg/dL -> mmol/L
            sham_sequence = np.array(glucose_data['sham']) * 0.0555
            
            # 添加个体差异
            for sensitivity in [0.7, 0.9, 1.0, 1.1, 1.3]:
                # 根据血糖水平和个体敏感性调整刺激参数
                adjusted_params = self._adjust_params_for_glucose_and_sensitivity(
                    stim_params, sham_sequence, sensitivity
                )
                
                samples.append({
                    'input_glucose': sham_sequence,
                    'stim_params': adjusted_params,
                    'individual_sensitivity': sensitivity,
                    'study_type': 'healthy_postprandial',
                    'protocol': protocol_name,
                    'stimulation_intensity': self._calculate_stimulation_intensity(adjusted_params)
                })
        
        return samples
    
    def _adjust_params_for_glucose_and_sensitivity(self, base_params, glucose_sequence, sensitivity):
        """
        根据血糖水平和个体敏感性调整刺激参数
        """
        # 计算血糖统计特征
        mean_glucose = np.mean(glucose_sequence)
        glucose_variance = np.var(glucose_sequence)
        glucose_trend = np.polyfit(range(len(glucose_sequence)), glucose_sequence, 1)[0]
        
        # 复制基础参数
        adjusted_params = base_params.copy()
        
        # 根据血糖水平调整频率
        if mean_glucose > 15:  # 高血糖
            adjusted_params[0] *= 1.2  # 增加频率
        elif mean_glucose < 8:  # 低血糖
            adjusted_params[0] *= 0.8  # 降低频率
        
        # 根据血糖变化趋势调整电流
        if glucose_trend > 0.1:  # 血糖上升趋势
            adjusted_params[1] *= 1.1  # 增加电流
        elif glucose_trend < -0.1:  # 血糖下降趋势
            adjusted_params[1] *= 0.9  # 降低电流
        
        # 根据血糖波动调整刺激时长
        if glucose_variance > 5:  # 血糖波动大
            adjusted_params[2] *= 1.15  # 增加刺激时长
        else:
            adjusted_params[2] *= 0.95  # 减少刺激时长
        
        # 根据个体敏感性调整脉宽
        adjusted_params[3] *= sensitivity
        
        # 根据血糖水平调整治疗周期
        if mean_glucose > 12:
            adjusted_params[4] *= 1.2  # 增加治疗周期
        elif mean_glucose < 6:
            adjusted_params[4] *= 0.8  # 减少治疗周期
        
        return adjusted_params
    
    def _create_alternating_frequency_params(self, base_freq_low, base_freq_high, glucose_sequence, sensitivity):
        """
        创建交替频率刺激参数
        根据血糖水平和个体敏感性调整2/15 Hz交替模式
        """
        # 计算血糖统计特征
        mean_glucose = np.mean(glucose_sequence)
        glucose_variance = np.var(glucose_sequence)
        glucose_trend = np.polyfit(range(len(glucose_sequence)), glucose_sequence, 1)[0]
        
        # 基础交替频率参数
        freq_low = base_freq_low
        freq_high = base_freq_high
        
        # 根据血糖水平调整交替频率
        if mean_glucose > 15:  # 高血糖 - 增加频率
            freq_low *= 1.3
            freq_high *= 1.3
        elif mean_glucose < 8:  # 低血糖 - 降低频率
            freq_low *= 0.7
            freq_high *= 0.7
        
        # 根据血糖趋势微调
        if glucose_trend > 0.1:  # 血糖上升趋势
            freq_low *= 1.1
            freq_high *= 1.1
        elif glucose_trend < -0.1:  # 血糖下降趋势
            freq_low *= 0.9
            freq_high *= 0.9
        
        # 确保频率在合理范围内
        freq_low = np.clip(freq_low, 0.5, 10.0)
        freq_high = np.clip(freq_high, 5.0, 30.0)
        
        # 返回平均频率作为主要参数
        avg_frequency = (freq_low + freq_high) / 2
        
        return avg_frequency, freq_low, freq_high
    
    def _generate_hourly_sequence(self, base_value, variation_type):
        """
        基于基础值生成1小时的血糖序列
        """
        sequence = np.zeros(12)
        
        if variation_type == 'high_variation':
            # 高血糖状态，变化较大
            for i in range(12):
                time_factor = 1 + 0.1 * np.sin(i * np.pi / 6)  # 周期性变化
                noise = np.random.normal(0, base_value * 0.05)
                sequence[i] = base_value * time_factor + noise
        
        elif variation_type == 'low_variation':
            # 治疗后状态，变化较小
            for i in range(12):
                time_factor = 1 + 0.05 * np.sin(i * np.pi / 6)
                noise = np.random.normal(0, base_value * 0.02)
                sequence[i] = base_value * time_factor + noise
        
        return np.clip(sequence, 3.0, 30.0)
    
    def _generate_postprandial_curve(self, peak_value, pattern_type):
        """
        生成餐后血糖曲线
        """
        sequence = np.zeros(12)
        
        if pattern_type == 'IGT_pattern':
            # IGT患者的餐后血糖模式
            baseline = peak_value * 0.7
            
            # 餐后血糖上升和下降模式
            time_points = np.linspace(0, 2, 12)  # 2小时
            for i, t in enumerate(time_points):
                if t <= 0.5:  # 前30分钟上升
                    sequence[i] = baseline + (peak_value - baseline) * (t / 0.5)
                elif t <= 1.0:  # 30-60分钟达到峰值
                    sequence[i] = peak_value
                else:  # 60-120分钟下降
                    sequence[i] = peak_value - (peak_value - baseline) * ((t - 1.0) / 1.0) * 0.8
        
        return np.clip(sequence, 3.0, 20.0)
    
    def _extend_to_12_points(self, glucose_values):
        """
        将血糖值扩展到12个点
        """
        if len(glucose_values) >= 12:
            return np.array(glucose_values[:12])
        
        # 使用插值扩展
        indices = np.linspace(0, len(glucose_values)-1, 12)
        extended = np.interp(indices, range(len(glucose_values)), glucose_values)
        
        return extended
    
    def _calculate_stimulation_intensity(self, stim_params):
        """
        计算刺激强度
        """
        freq, amp, duration, pulse_width, session_duration = stim_params
        
        # 基础强度计算
        base_intensity = (freq * amp * duration * pulse_width) / 1000000
        
        # 考虑治疗周期的累积效应
        if session_duration >= 6:
            base_intensity *= 1.3
        elif session_duration >= 2:
            base_intensity *= 1.1
        
        return base_intensity
    
    def normalize_data(self, samples):
        """
        标准化数据
        """
        # 提取所有血糖序列
        all_glucose = np.array([sample['input_glucose'] for sample in samples])
        
        # 标准化血糖数据
        normalized_glucose = self.glucose_scaler.fit_transform(all_glucose)
        
        # 标准化刺激参数
        all_params = np.array([sample['stim_params'] for sample in samples])
        normalized_params = self.param_scaler.fit_transform(all_params)
        
        # 更新样本数据
        for i, sample in enumerate(samples):
            sample['input_glucose_norm'] = normalized_glucose[i]
            sample['stim_params_norm'] = normalized_params[i]
            
        return samples
    
    def inverse_transform_glucose(self, normalized_glucose):
        """
        反标准化血糖数据
        """
        return self.glucose_scaler.inverse_transform(normalized_glucose)
    
    def inverse_transform_params(self, normalized_params):
        """
        反标准化刺激参数
        """
        return self.param_scaler.inverse_transform(normalized_params)
    
    def get_paper_summary(self):
        """
        获取论文数据总结
        """
        summary = {
            'paper1_zdf_mice': {
                'study_duration': '5 weeks',
                'stimulation': '2/15 Hz alternating (2Hz↔15Hz), 2mA, 30min',
                'glucose_reduction': 'From 18-28 to 10 mmol/L',
                'sample_count': len(self._create_zdf_mice_samples())
            },
            'paper2_human_igt': {
                'study_duration': '12 weeks',
                'stimulation': '20 Hz, 1mA, 20min, 1ms pulse',
                'glucose_reduction': '2hPG: 9.7→7.3 mmol/L',
                'sample_count': len(self._create_human_igt_samples())
            },
            'paper3_healthy_postprandial': {
                'study_duration': 'Acute (single session)',
                'stimulation': '10 Hz, 2.3mA, 30min, 0.3ms pulse',
                'effect': 'Postprandial glucose suppression',
                'sample_count': len(self._create_healthy_postprandial_samples())
            }
        }
        
        return summary

class taVNSDataset(Dataset):
    """
    taVNS数据集
    专门用于taVNS参数预测
    """
    
    def __init__(self, samples, sequence_length=12):
        self.samples = samples
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 输入：血糖序列
        input_glucose = torch.FloatTensor(sample['input_glucose_norm'])
        
        # 输出：刺激参数
        stim_params = torch.FloatTensor(sample['stim_params_norm'])
        
        return {
            'input_glucose': input_glucose,
            'stim_params': stim_params,
            'individual_sensitivity': sample['individual_sensitivity'],
            'stimulation_intensity': sample['stimulation_intensity'],
            'study_type': sample['study_type']
        }

def create_data_loaders(samples, batch_size=16, train_ratio=0.8):
    """
    创建训练和验证数据加载器
    """
    # 随机打乱样本
    np.random.shuffle(samples)
    
    # 划分训练集和验证集
    train_size = int(train_ratio * len(samples))
    val_size = len(samples) - train_size
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    # 创建数据集
    train_dataset = taVNSDataset(train_samples)
    val_dataset = taVNSDataset(val_samples)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 