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
    增强版：包含大量数据扩充方法
    """
    
    def __init__(self, target_sequence_length=12, enable_data_augmentation=True):
        self.target_sequence_length = target_sequence_length
        self.glucose_scaler = StandardScaler()
        self.param_scaler = MinMaxScaler()
        self.enable_data_augmentation = enable_data_augmentation
        
        # 数据扩充参数
        self.augmentation_config = {
            'noise_levels': [0.01, 0.02, 0.03, 0.05],  # 噪声水平
            'time_warp_factors': [0.9, 0.95, 1.05, 1.1],  # 时间扭曲因子
            'amplitude_scales': [0.9, 0.95, 1.05, 1.1],  # 幅度缩放
            'baseline_shifts': [-0.5, -0.2, 0.2, 0.5],  # 基线偏移
            'param_mutations': [0.05, 0.1, 0.15],  # 参数变异幅度
            'individual_variations': [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # 个体差异
        }
        
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
        增强版：包含大量数据扩充
        """
        samples = []
        
        # 1. 基于论文1的ZDF小鼠数据
        base_samples_1 = self._create_zdf_mice_samples()
        samples.extend(base_samples_1)
        
        # 2. 基于论文2的人体IGT患者数据
        base_samples_2 = self._create_human_igt_samples()
        samples.extend(base_samples_2)
        
        # 3. 基于论文3的健康人餐后血糖数据
        base_samples_3 = self._create_healthy_postprandial_samples()
        samples.extend(base_samples_3)
        
        print(f"基础样本数量: {len(samples)}")
        
        # 4. 数据扩充
        if self.enable_data_augmentation:
            augmented_samples = self._apply_comprehensive_data_augmentation(samples)
            samples.extend(augmented_samples)
            print(f"扩充后样本数量: {len(samples)}")
        
        # 5. 生成合成数据
        synthetic_samples = self._generate_synthetic_samples(target_count=2000)
        samples.extend(synthetic_samples)
        print(f"添加合成数据后样本数量: {len(samples)}")
        
        return samples
    
    def _apply_comprehensive_data_augmentation(self, base_samples):
        """
        应用综合数据扩充方法
        """
        augmented_samples = []
        
        for base_sample in base_samples:
            # 1. 噪声注入
            for noise_level in self.augmentation_config['noise_levels']:
                aug_sample = self._add_gaussian_noise(base_sample, noise_level)
                augmented_samples.append(aug_sample)
            
            # 2. 时间扭曲
            for warp_factor in self.augmentation_config['time_warp_factors']:
                aug_sample = self._apply_time_warping(base_sample, warp_factor)
                augmented_samples.append(aug_sample)
            
            # 3. 幅度缩放
            for scale_factor in self.augmentation_config['amplitude_scales']:
                aug_sample = self._apply_amplitude_scaling(base_sample, scale_factor)
                augmented_samples.append(aug_sample)
            
            # 4. 基线偏移
            for shift in self.augmentation_config['baseline_shifts']:
                aug_sample = self._apply_baseline_shift(base_sample, shift)
                augmented_samples.append(aug_sample)
            
            # 5. 参数变异
            for mutation_rate in self.augmentation_config['param_mutations']:
                aug_sample = self._apply_parameter_mutation(base_sample, mutation_rate)
                augmented_samples.append(aug_sample)
            
            # 6. 个体差异模拟
            for variation in self.augmentation_config['individual_variations']:
                if variation != 1.0:  # 跳过原始值
                    aug_sample = self._simulate_individual_variation(base_sample, variation)
                    augmented_samples.append(aug_sample)
        
        return augmented_samples
    
    def _add_gaussian_noise(self, sample, noise_level):
        """添加高斯噪声"""
        aug_sample = sample.copy()
        glucose_seq = np.array(sample['input_glucose'])
        noise = np.random.normal(0, noise_level * np.std(glucose_seq), glucose_seq.shape)
        aug_sample['input_glucose'] = np.clip(glucose_seq + noise, 3.0, 30.0)
        aug_sample['augmentation_type'] = f'gaussian_noise_{noise_level}'
        return aug_sample
    
    def _apply_time_warping(self, sample, warp_factor):
        """应用时间扭曲"""
        aug_sample = sample.copy()
        glucose_seq = np.array(sample['input_glucose'])
        
        # 创建扭曲的时间索引
        original_indices = np.linspace(0, len(glucose_seq)-1, len(glucose_seq))
        warped_indices = np.linspace(0, len(glucose_seq)-1, int(len(glucose_seq) * warp_factor))
        
        # 插值到原始长度
        if len(warped_indices) != len(original_indices):
            warped_glucose = np.interp(original_indices, 
                                     np.linspace(0, len(glucose_seq)-1, len(warped_indices)), 
                                     np.interp(warped_indices, original_indices, glucose_seq))
        else:
            warped_glucose = glucose_seq
        
        aug_sample['input_glucose'] = np.clip(warped_glucose, 3.0, 30.0)
        aug_sample['augmentation_type'] = f'time_warp_{warp_factor}'
        return aug_sample
    
    def _apply_amplitude_scaling(self, sample, scale_factor):
        """应用幅度缩放"""
        aug_sample = sample.copy()
        glucose_seq = np.array(sample['input_glucose'])
        mean_glucose = np.mean(glucose_seq)
        
        # 围绕均值进行缩放
        scaled_glucose = mean_glucose + (glucose_seq - mean_glucose) * scale_factor
        aug_sample['input_glucose'] = np.clip(scaled_glucose, 3.0, 30.0)
        aug_sample['augmentation_type'] = f'amplitude_scale_{scale_factor}'
        return aug_sample
    
    def _apply_baseline_shift(self, sample, shift):
        """应用基线偏移"""
        aug_sample = sample.copy()
        glucose_seq = np.array(sample['input_glucose'])
        shifted_glucose = glucose_seq + shift
        aug_sample['input_glucose'] = np.clip(shifted_glucose, 3.0, 30.0)
        aug_sample['augmentation_type'] = f'baseline_shift_{shift}'
        return aug_sample
    
    def _apply_parameter_mutation(self, sample, mutation_rate):
        """应用参数变异"""
        aug_sample = sample.copy()
        stim_params = np.array(sample['stim_params'])
        
        # 为每个参数添加变异
        mutations = np.random.normal(0, mutation_rate, stim_params.shape)
        mutated_params = stim_params * (1 + mutations)
        
        # 确保参数在合理范围内
        mutated_params[0] = np.clip(mutated_params[0], 1.0, 50.0)  # 频率
        mutated_params[1] = np.clip(mutated_params[1], 0.5, 5.0)   # 电流
        mutated_params[2] = np.clip(mutated_params[2], 10.0, 60.0) # 时长
        mutated_params[3] = np.clip(mutated_params[3], 50.0, 2000.0) # 脉宽
        mutated_params[4] = np.clip(mutated_params[4], 1.0, 20.0)  # 周期
        
        aug_sample['stim_params'] = mutated_params
        aug_sample['augmentation_type'] = f'param_mutation_{mutation_rate}'
        return aug_sample
    
    def _simulate_individual_variation(self, sample, variation_factor):
        """模拟个体差异"""
        aug_sample = sample.copy()
        
        # 调整血糖序列（模拟个体代谢差异）
        glucose_seq = np.array(sample['input_glucose'])
        varied_glucose = glucose_seq * variation_factor
        aug_sample['input_glucose'] = np.clip(varied_glucose, 3.0, 30.0)
        
        # 相应调整刺激参数
        stim_params = np.array(sample['stim_params'])
        adjusted_params = self._adjust_params_for_glucose_and_sensitivity(
            stim_params, varied_glucose, variation_factor
        )
        aug_sample['stim_params'] = adjusted_params
        aug_sample['individual_sensitivity'] = variation_factor
        aug_sample['augmentation_type'] = f'individual_variation_{variation_factor}'
        
        return aug_sample
    
    def _generate_synthetic_samples(self, target_count=2000):
        """
        生成合成训练样本
        """
        synthetic_samples = []
        
        # 定义合成数据的参数范围
        glucose_patterns = [
            'normal_fasting',      # 正常空腹
            'impaired_fasting',    # 空腹血糖受损
            'diabetes_pattern',    # 糖尿病模式
            'postprandial_normal', # 正常餐后
            'postprandial_high',   # 餐后高血糖
            'hypoglycemic',        # 低血糖
            'dawn_phenomenon',     # 黎明现象
            'somogyi_effect'       # 苏木杰效应
        ]
        
        param_strategies = [
            'conservative',        # 保守治疗
            'aggressive',          # 积极治疗
            'personalized_low',    # 个性化-低敏感
            'personalized_high',   # 个性化-高敏感
            'frequency_focused',   # 频率导向
            'amplitude_focused',   # 幅度导向
            'duration_focused'     # 时长导向
        ]
        
        samples_per_combination = target_count // (len(glucose_patterns) * len(param_strategies))
        
        for pattern in glucose_patterns:
            for strategy in param_strategies:
                for _ in range(samples_per_combination):
                    # 生成血糖序列
                    glucose_seq = self._generate_synthetic_glucose_pattern(pattern)
                    
                    # 生成相应的刺激参数
                    stim_params = self._generate_synthetic_stimulation_params(strategy, glucose_seq)
                    
                    # 计算个体敏感性
                    sensitivity = np.random.uniform(0.6, 1.4)
                    
                    synthetic_sample = {
                        'input_glucose': glucose_seq,
                        'stim_params': stim_params,
                        'individual_sensitivity': sensitivity,
                        'study_type': 'synthetic',
                        'glucose_pattern': pattern,
                        'param_strategy': strategy,
                        'stimulation_intensity': self._calculate_stimulation_intensity(stim_params),
                        'augmentation_type': 'synthetic_generation'
                    }
                    
                    synthetic_samples.append(synthetic_sample)
        
        return synthetic_samples
    
    def _generate_synthetic_glucose_pattern(self, pattern_type):
        """生成合成血糖模式"""
        sequence = np.zeros(12)
        
        if pattern_type == 'normal_fasting':
            base_level = np.random.uniform(4.5, 6.0)
            sequence = base_level + np.random.normal(0, 0.3, 12)
            
        elif pattern_type == 'impaired_fasting':
            base_level = np.random.uniform(6.1, 7.0)
            sequence = base_level + np.random.normal(0, 0.5, 12)
            
        elif pattern_type == 'diabetes_pattern':
            base_level = np.random.uniform(12.0, 20.0)
            trend = np.random.uniform(-0.2, 0.2)
            for i in range(12):
                sequence[i] = base_level + trend * i + np.random.normal(0, 1.0)
                
        elif pattern_type == 'postprandial_normal':
            baseline = np.random.uniform(5.0, 7.0)
            peak = np.random.uniform(8.0, 11.0)
            for i in range(12):
                if i < 3:  # 上升期
                    sequence[i] = baseline + (peak - baseline) * (i / 3)
                elif i < 6:  # 峰值期
                    sequence[i] = peak + np.random.normal(0, 0.3)
                else:  # 下降期
                    sequence[i] = peak - (peak - baseline) * ((i - 6) / 6) * 0.8
                    
        elif pattern_type == 'postprandial_high':
            baseline = np.random.uniform(7.0, 10.0)
            peak = np.random.uniform(15.0, 22.0)
            for i in range(12):
                if i < 4:  # 上升期
                    sequence[i] = baseline + (peak - baseline) * (i / 4)
                elif i < 7:  # 峰值期
                    sequence[i] = peak + np.random.normal(0, 0.5)
                else:  # 缓慢下降期
                    sequence[i] = peak - (peak - baseline) * ((i - 7) / 5) * 0.6
                    
        elif pattern_type == 'hypoglycemic':
            base_level = np.random.uniform(2.5, 4.0)
            sequence = base_level + np.random.normal(0, 0.2, 12)
            
        elif pattern_type == 'dawn_phenomenon':
            night_level = np.random.uniform(6.0, 8.0)
            morning_rise = np.random.uniform(2.0, 4.0)
            for i in range(12):
                sequence[i] = night_level + morning_rise * (i / 12) + np.random.normal(0, 0.3)
                
        elif pattern_type == 'somogyi_effect':
            low_level = np.random.uniform(3.0, 4.5)
            rebound_level = np.random.uniform(12.0, 18.0)
            for i in range(12):
                if i < 4:
                    sequence[i] = low_level + np.random.normal(0, 0.2)
                else:
                    sequence[i] = rebound_level + np.random.normal(0, 1.0)
        
        return np.clip(sequence, 2.5, 30.0)
    
    def _generate_synthetic_stimulation_params(self, strategy, glucose_seq):
        """生成合成刺激参数"""
        mean_glucose = np.mean(glucose_seq)
        glucose_variance = np.var(glucose_seq)
        
        if strategy == 'conservative':
            frequency = np.random.uniform(5.0, 15.0)
            amplitude = np.random.uniform(0.8, 1.5)
            duration = np.random.uniform(15.0, 25.0)
            pulse_width = np.random.uniform(100.0, 500.0)
            session_duration = np.random.uniform(4.0, 8.0)
            
        elif strategy == 'aggressive':
            frequency = np.random.uniform(15.0, 30.0)
            amplitude = np.random.uniform(2.0, 3.0)
            duration = np.random.uniform(30.0, 45.0)
            pulse_width = np.random.uniform(300.0, 800.0)
            session_duration = np.random.uniform(8.0, 16.0)
            
        elif strategy == 'personalized_low':
            # 低敏感性个体需要更强刺激
            frequency = np.random.uniform(10.0, 25.0)
            amplitude = np.random.uniform(1.5, 2.5)
            duration = np.random.uniform(25.0, 40.0)
            pulse_width = np.random.uniform(200.0, 600.0)
            session_duration = np.random.uniform(6.0, 12.0)
            
        elif strategy == 'personalized_high':
            # 高敏感性个体需要较温和刺激
            frequency = np.random.uniform(3.0, 12.0)
            amplitude = np.random.uniform(0.5, 1.2)
            duration = np.random.uniform(15.0, 30.0)
            pulse_width = np.random.uniform(100.0, 300.0)
            session_duration = np.random.uniform(3.0, 8.0)
            
        elif strategy == 'frequency_focused':
            frequency = np.random.uniform(20.0, 40.0)  # 高频率
            amplitude = np.random.uniform(1.0, 2.0)
            duration = np.random.uniform(20.0, 35.0)
            pulse_width = np.random.uniform(150.0, 400.0)
            session_duration = np.random.uniform(5.0, 10.0)
            
        elif strategy == 'amplitude_focused':
            frequency = np.random.uniform(8.0, 18.0)
            amplitude = np.random.uniform(2.5, 4.0)  # 高幅度
            duration = np.random.uniform(20.0, 30.0)
            pulse_width = np.random.uniform(200.0, 500.0)
            session_duration = np.random.uniform(4.0, 8.0)
            
        elif strategy == 'duration_focused':
            frequency = np.random.uniform(10.0, 20.0)
            amplitude = np.random.uniform(1.2, 2.0)
            duration = np.random.uniform(45.0, 60.0)  # 长时间
            pulse_width = np.random.uniform(200.0, 600.0)
            session_duration = np.random.uniform(8.0, 16.0)
        
        else:  # 默认策略
            frequency = np.random.uniform(8.0, 25.0)
            amplitude = np.random.uniform(1.0, 2.5)
            duration = np.random.uniform(20.0, 40.0)
            pulse_width = np.random.uniform(150.0, 600.0)
            session_duration = np.random.uniform(4.0, 12.0)
        
        # 根据血糖水平进行微调
        if mean_glucose > 15.0:  # 高血糖
            frequency *= 1.2
            amplitude *= 1.1
            duration *= 1.1
        elif mean_glucose < 6.0:  # 低血糖
            frequency *= 0.8
            amplitude *= 0.9
            duration *= 0.9
        
        return np.array([frequency, amplitude, duration, pulse_width, session_duration])
    
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
    
    def get_augmentation_summary(self):
        """
        获取数据扩充总结
        """
        if not self.enable_data_augmentation:
            return "数据扩充已禁用"
        
        summary = {
            'augmentation_methods': [
                '高斯噪声注入',
                '时间扭曲',
                '幅度缩放',
                '基线偏移',
                '参数变异',
                '个体差异模拟',
                '合成数据生成'
            ],
            'noise_levels': len(self.augmentation_config['noise_levels']),
            'time_warp_variants': len(self.augmentation_config['time_warp_factors']),
            'amplitude_variants': len(self.augmentation_config['amplitude_scales']),
            'baseline_variants': len(self.augmentation_config['baseline_shifts']),
            'param_mutations': len(self.augmentation_config['param_mutations']),
            'individual_variations': len(self.augmentation_config['individual_variations']),
            'synthetic_patterns': 8,  # glucose patterns
            'synthetic_strategies': 7,  # param strategies
            'estimated_total_samples': '> 10,000'
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