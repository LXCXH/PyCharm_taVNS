#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

/*
 * 数据标准化参数
 * 生成时间: 2025-07-28 20:00:06
 * 参数来源: 训练数据 (来源: Training_Outputs\training_output_20250728_195138)
 * 
 * 此文件由generate_arduino_files.py自动生成
 * 请勿手动修改，重新生成时会覆盖
 */

// =============================================================================
// 血糖数据标准化参数 (StandardScaler)
// =============================================================================

// 血糖序列的均值 (用于标准化: (x - mean) / std)
const float glucose_scaler_mean[12] = {
  9.828118, 10.264870, 10.885009, 11.528777, 12.584801, 12.561557, 12.403735, 12.182517, 11.780627, 11.487507, 11.460243, 11.331880
};

// 血糖序列的标准差 (用于标准化: (x - mean) / std)
const float glucose_scaler_std[12] = {
  7.657434, 7.467002, 7.682013, 8.055747, 7.812631, 7.697483, 7.490601, 7.331663, 6.914463, 6.689632, 6.970897, 7.146074
};

// =============================================================================
// 参数数据标准化参数 (MinMaxScaler)
// =============================================================================

// 参数的最小值 (用于反标准化: x * (max - min) + min)
const float param_scaler_min[5] = {
  // 顺序: [频率, 电流, 时长, 脉宽, 周期]
  2.471778, 0.478536, 13.614690, 98.000000, 2.560000
};

// 参数的最大值 (用于反标准化: x * (max - min) + min)
const float param_scaler_max[5] = {
  // 顺序: [频率, 电流, 时长, 脉宽, 周期]
  47.749519, 4.370435, 65.915754, 1820.000000, 15.997578
};

// =============================================================================
// 参数名称和单位 (用于显示)
// =============================================================================

const char* param_names[5] = {
  "频率", "电流", "时长", "脉宽", "周期"
};

const char* param_units[5] = {
  "Hz", "mA", "min", "μs", "周"
};

// =============================================================================
// 辅助函数
// =============================================================================

// 血糖数据标准化函数
inline void normalize_glucose_value(float raw_value, int index, float* normalized_value) {
  *normalized_value = (raw_value - glucose_scaler_mean[index]) / glucose_scaler_std[index];
}

// 参数数据反标准化函数
inline void denormalize_param_value(float normalized_value, int index, float* param_value) {
  *param_value = normalized_value * (param_scaler_max[index] - param_scaler_min[index]) + param_scaler_min[index];
}

// 验证参数范围是否合理
inline bool validate_param_range(float param_value, int index) {
  return (param_value >= param_scaler_min[index] && param_value <= param_scaler_max[index]);
}

#endif // SCALER_PARAMS_H