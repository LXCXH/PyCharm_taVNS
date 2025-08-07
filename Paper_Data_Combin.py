#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# —— 修复中文字体 & 负号显示 ——
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 根据本机可用字体调整
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 通用函数
# ============================================================================

def simulate_g_series(mean, n=12, noise_std=0.05):
    """
    生成长度为 n 的带噪声序列，均值为 mean，
    最后校正回 mean。
    """
    noise = np.random.normal(0, noise_std * mean, size=n)
    series = mean + noise
    series -= (series.mean() - mean)
    return series

def sample_subject_mean(median, q1, q3, min_val, max_val):
    """
    首先为每个受试者从截断正态分布中采样一个"日均值"：
    median, IQR->sigma, truncated to [min_val, max_val].
    """
    sigma = (q3 - q1) / 1.349
    while True:
        m = np.random.normal(loc=median, scale=sigma)
        if min_val <= m <= max_val:
            return m

# ============================================================================
# Paper 1 处理
# ============================================================================

def process_paper1():
    print("=" * 60)
    print("处理 Paper 1 数据...")
    print("=" * 60)
    
    # —— 原始论文数据 ——
    paper1_data = {
        'stimulation_params': {
            'frequency_low': 2,    # Hz
            'frequency_high': 15,  # Hz
            'amplitude': 2.0,      # mA
            'duration': 30,        # minutes per session
            'pulse_width': 200,    # µs (假设)
            'session_weeks': 6     # 共 6 周
        },
        'baseline_glucose': {
            'control':   [19, 22, 24, 27, 29, 32],  # 对照组周均值
            'treatment': [19, 10, 11, 12, 12, 11]   # 实验组周均值
        }
    }

    # —— 参数准备 ——
    params = paper1_data['stimulation_params']
    freq_eq = (params['frequency_low'] + params['frequency_high']) / 2.0
    # 小鼠→人体 映射系数：假设人体空腹平均 7.0 mmol/L 对应小鼠 baseline control Week0=19
    mapping_factor = 7.0 / paper1_data['baseline_glucose']['control'][0]

    # —— 1) Baseline: 从对照组末尾 (32) 线性下降到实验组末尾 (11)，生成 42 天均值 ——
    weeks = params['session_weeks']
    days = weeks * 7                 # 42 天
    start, end = paper1_data['baseline_glucose']['control'][-1], paper1_data['baseline_glucose']['treatment'][-1]
    daily_means = np.linspace(start, end, days)

    # —— 2) 模拟 42 条 g[12]，并映射到人体水平 ——
    baseline_sim = np.array([simulate_g_series(m) for m in daily_means])
    baseline_sim_human = baseline_sim * mapping_factor

    # —— 3) 构建 DataFrame 并保存 CSV ——
    cols = [f'g{i+1}' for i in range(12)]
    df = pd.DataFrame(baseline_sim_human, columns=cols)
    df['frequency_Hz'] = freq_eq
    df['amplitude_mA'] = params['amplitude']
    df['duration_min'] = params['duration']
    df['pulse_width_us'] = params['pulse_width']
    df['session_idx'] = np.arange(0, days)

    df.to_csv("Training_Dataset/paper1_42.csv", index=False)
    print("已保存：Training_Dataset/paper1_42.csv")

    # —— 4) 绘图：Baseline 42 条曲线 ——
    x = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, days))

    for idx, series in enumerate(baseline_sim_human):
        ax.plot(x, series, color=colors[idx], linewidth=1)

    ax.set_xlabel('Reading Index (每5分钟一次)')
    ax.set_ylabel('Glucose (mmol/L)')
    ax.set_title('Baseline 处理组模拟血糖曲线（42 条，单调下降）')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, days))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[1, 14, 28, 42])
    cbar.set_label('Day Index')

    plt.tight_layout()
    plt.show()

# ============================================================================
# Paper 2 处理
# ============================================================================

def process_paper2():
    print("=" * 60)
    print("处理 Paper 2 数据...")
    print("=" * 60)
    
    # —— 论文2 数据 ——
    paper2_data = {
        'study_type': 'human_IGT',
        'stimulation_params': {
            'frequency': 20,       # Hz
            'amplitude': 1.0,      # mA
            'duration': 20,        # min
            'pulse_width': 1000,   # µs
            'session_duration': 12 # 周
        },
        '2hPG_results': {
            'baseline':      {'ta_vns': 9.7, 'sham': 9.1, 'no_treatment': 9.3},
            '6_weeks':       {'ta_vns': 7.3, 'sham': 8.0, 'no_treatment': 9.5},
            '12_weeks':      {'ta_vns': 7.5, 'sham': 8.0, 'no_treatment': 10.0}
        }
    }

    # —— 参数提取 ——
    params = paper2_data['stimulation_params']
    upper_start = paper2_data['2hPG_results']['baseline']['no_treatment']
    lower_end = paper2_data['2hPG_results']['12_weeks']['ta_vns']
    weeks = params['session_duration']
    days = weeks * 7                # 12 周 × 7 天 = 84 天
    sessions_per_day = 2
    total_sessions = days * sessions_per_day  # 168 条序列

    freq = params['frequency']
    amplitude = params['amplitude']
    duration_min = params['duration']
    pulse_width_us = params['pulse_width']

    # —— 生成每日均值单调线性下降 ——
    daily_means = np.linspace(upper_start, lower_end, days)

    # —— 为每天两次会话复制均值，并模拟 168 条序列 ——
    session_means = np.repeat(daily_means, sessions_per_day)
    all_series = np.array([simulate_g_series(m) for m in session_means])

    # —— 构建 DataFrame 并保存 CSV ——
    cols = [f'g{i+1}' for i in range(12)]
    df = pd.DataFrame(all_series, columns=cols)
    df['frequency_Hz'] = freq
    df['amplitude_mA'] = amplitude
    df['duration_min'] = duration_min
    df['pulse_width_us'] = pulse_width_us
    df['session_idx'] = np.arange(0, total_sessions)

    df.to_csv("Training_Dataset/paper2_168.csv", index=False)
    print("已保存：Training_Dataset/paper2_168.csv")

    # —— 绘图：168 条会话曲线 ——
    x = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(8,5))
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0,1,total_sessions))

    for idx, series in enumerate(all_series):
        ax.plot(x, series, color=colors[idx], linewidth=1)

    ax.set_xlabel('Reading Index (每5分钟一次)')
    ax.set_ylabel('2hPG Simulated (mmol/L)')
    ax.set_title('Human IGT taVNS 2hPG 模拟曲线（168 会话，单调下降）')

    # Colorbar 标注第 1／42／84／126／168 会话
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, total_sessions))
    sm.set_array([])
    ticks = [1, 42, 84, 126, 168]
    cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
    cbar.set_label('Session Index')

    plt.tight_layout()
    plt.show()

# ============================================================================
# Paper 3 处理
# ============================================================================

def process_paper3():
    print("=" * 60)
    print("处理 Paper 3 数据...")
    print("=" * 60)
    
    # —— 论文3 数据 ——
    paper3_data = {
        'stimulation_params': {
            'protocol1': {'frequency': 10, 'amplitude': 2.3, 'duration': 30, 'pulse_width': 300, 'n_subjects': 16},
            'protocol2': {'frequency': 10, 'amplitude': 2.3, 'duration': 30, 'pulse_width': 300, 'n_subjects': 10}
        },
        'glucose_stats': {
            'protocol1': {
                'control': {
                    'before': {'median': 104, 'q1': 97, 'q3': 111, 'min': 86, 'max': 137},
                    'after': {'median': 97, 'q1': 92, 'q3': 106, 'min': 82, 'max': 137}
                },
                'treatment': {
                    'before': {'median': 99, 'q1': 96, 'q3': 105, 'min': 86, 'max': 116},
                    'after': {'median': 93, 'q1': 89, 'q3': 97, 'min': 84, 'max': 108}
                }
            },
            'protocol2': {
                'control': {
                    'before': {'median': 95, 'q1': 88, 'q3': 103, 'min': 80, 'max': 112},
                    'after': {'median': 146, 'q1': 123, 'q3': 174, 'min': 97, 'max': 212}
                },
                'treatment': {
                    'before': {'median': 93, 'q1': 89, 'q3': 104, 'min': 86, 'max': 122},
                    'after': {'median': 162, 'q1': 124, 'q3': 176, 'min': 96, 'max': 224}
                }
            }
        }
    }

    records = []
    # 按 protocol -> group -> subject -> session 顺序
    for prot in ['protocol1', 'protocol2']:
        pm = paper3_data['stimulation_params'][prot]
        stats = paper3_data['glucose_stats'][prot]
        n_subj = pm['n_subjects']
        for group in ['control', 'treatment']:
            for subj in range(1, n_subj + 1):
                for session_idx, session in enumerate(['before', 'after']):
                    st = stats[group][session]
                    # 第一步：为该受试者采样一个日均血糖
                    subj_mean = sample_subject_mean(
                        st['median'], st['q1'], st['q3'], st['min'], st['max']
                    )
                    # 第二步：基于该均值，生成 12 点读数序列
                    series = simulate_g_series(subj_mean, n=12, noise_std=0.05)

                    # 记录
                    rec = {f'g{i + 1}': float(series[i]) for i in range(12)}
                    rec.update({
                        'frequency_Hz': pm['frequency'],
                        'amplitude_mA': pm['amplitude'],
                        'duration_min': pm['duration'],
                        'pulse_width_us': pm['pulse_width'],
                        'session_idx': session_idx,
                        'protocol': prot,
                        'group': group,
                        'subject': subj
                    })
                    records.append(rec)

    # 构造 DataFrame 并按指定列顺序输出
    df = pd.DataFrame(records)
    columns = [f'g{i}' for i in range(1, 13)] + [
        'frequency_Hz', 'amplitude_mA', 'duration_min', 'pulse_width_us', 'session_idx'
    ]
    df_output = df[columns]
    df_output.to_csv("Training_Dataset/paper3_104.csv", index=False)
    print("已保存：Training_Dataset/paper3_104.csv")

    # —— 绘图：分别打印 protocol1/2 下 control/treatment before/after 的所有个体曲线 ——
    x = np.arange(1, 13)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    color_map = {
        ('control', 0): 'C0',      # control-before
        ('control', 1): 'C1',      # control-after
        ('treatment', 0): 'C2',    # treatment-before
        ('treatment', 1): 'C3'     # treatment-after
    }

    for ax, prot in zip(axes, ['protocol1', 'protocol2']):
        # 筛选当前协议的数据
        prot_data = df[df['protocol'] == prot]
        curve_count = 0
        
        for _, row in prot_data.iterrows():
            y = [row[f'g{i}'] for i in range(1, 13)]
            color = color_map[(row['group'], row['session_idx'])]
            ax.plot(x, y, color=color, alpha=0.4, linewidth=1)
            curve_count += 1
        
        ax.set_title(f"{prot} 共 {curve_count} 条曲线")
        ax.set_ylabel('Glucose (mg/dL)')
        
        # 添加图例
        legend_lines = [plt.Line2D([0], [0], color=c, lw=2) for c in color_map.values()]
        legend_labels = [f"{g}-{['before', 'after'][s]}" for (g, s) in color_map.keys()]
        ax.legend(legend_lines, legend_labels, title='Group_Session')
    
    axes[-1].set_xlabel('Reading Index (1–12)')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：依次处理三篇论文的数据
    """
    print("\n" + "=" * 60)
    print("开始整合处理三篇论文数据")
    print("=" * 60 + "\n")
    
    # 处理 Paper 1
    process_paper1()
    print()
    
    # 处理 Paper 2
    process_paper2()
    print()
    
    # 处理 Paper 3
    process_paper3()
    
    print("\n" + "=" * 60)
    print("所有数据处理完成！")
    print("生成的文件：")
    print("  - Training_Dataset/paper1_42.csv")
    print("  - Training_Dataset/paper2_168.csv")
    print("  - Training_Dataset/paper3_104.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()