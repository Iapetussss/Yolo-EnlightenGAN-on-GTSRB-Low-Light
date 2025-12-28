#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 PPT 所需的所有图表和数据汇总
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

def load_experiment_results():
    """加载所有实验结果"""
    results = {}
    
    # GTSRB 实验
    gtsrb_experiments = {
        'Baseline': 'experiments/exp1_baseline/run8/results.csv',
        'Traditional': 'experiments/exp2_traditional/run/results.csv',
        'Mild': 'experiments/exp4_mild/run/results.csv',
        'EnlightenGAN': 'experiments/exp3_enlightengan/run/results.csv'
    }
    
    for name, path in gtsrb_experiments.items():
        csv_path = Path(path)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            last_row = df.iloc[-1]
            results[f'GTSRB_{name}'] = {
                'mAP@0.5': last_row['metrics/mAP50(B)'],
                'mAP@0.5:0.95': last_row['metrics/mAP50-95(B)'],
                'Precision': last_row['metrics/precision(B)'],
                'Recall': last_row['metrics/recall(B)']
            }
    
    # CNTSSS 实验
    cntsss_experiments = {
        'Baseline': 'experiments/cntsss_baseline/run/results.csv',
        'Mild': 'experiments/cntsss_mild_enhanced/run/results.csv',
        'EnlightenGAN': 'experiments/cntsss_enlightengan_mild_enhanced/run/results.csv'
    }
    
    for name, path in cntsss_experiments.items():
        csv_path = Path(path)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            last_row = df.iloc[-1]
            results[f'CNTSSS_{name}'] = {
                'mAP@0.5': last_row['metrics/mAP50(B)'],
                'mAP@0.5:0.95': last_row['metrics/mAP50-95(B)'],
                'Precision': last_row['metrics/precision(B)'],
                'Recall': last_row['metrics/recall(B)']
            }
    
    return results

def generate_comparison_charts(results, output_dir):
    """生成对比图表"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图1: GTSRB mAP 对比
    fig, ax = plt.subplots(figsize=(12, 7))
    
    gtsrb_methods = ['Baseline', 'Traditional', 'Mild', 'EnlightenGAN']
    gtsrb_map = [results.get(f'GTSRB_{m}', {}).get('mAP@0.5', 0) * 100 for m in gtsrb_methods]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(gtsrb_methods, gtsrb_map, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_ylabel('mAP@0.5 (%)', fontsize=14, fontweight='bold')
    ax.set_title('GTSRB Dataset - Enhancement Methods Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(gtsrb_map) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=70, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline Reference')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gtsrb_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ GTSRB 对比图")
    
    # 图2: CNTSSS mAP 对比
    fig, ax = plt.subplots(figsize=(10, 7))
    
    cntsss_methods = ['Baseline', 'Mild Enhancement', 'EnlightenGAN']
    cntsss_map = [results.get(f'CNTSSS_{m}', {}).get('mAP@0.5', 0) * 100 for m in ['Baseline', 'Mild', 'EnlightenGAN']]
    
    colors = ['#3498db', '#f39c12', '#e74c3c']
    bars = ax.bar(cntsss_methods, cntsss_map, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_ylabel('mAP@0.5 (%)', fontsize=14, fontweight='bold')
    ax.set_title('CNTSSS Dataset - Enhancement Methods Comparison\n(Real Night-Time Scenes)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(cntsss_map) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cntsss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ CNTSSS 对比图")
    
    # 图3: 多指标对比（GTSRB Baseline vs Best）
    fig, ax = plt.subplots(figsize=(10, 7))
    
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    baseline_values = [
        results['GTSRB_Baseline']['mAP@0.5'] * 100,
        results['GTSRB_Baseline']['mAP@0.5:0.95'] * 100,
        results['GTSRB_Baseline']['Precision'] * 100,
        results['GTSRB_Baseline']['Recall'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x, baseline_values, width, label='Baseline', color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title('GTSRB Baseline - Detailed Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gtsrb_metrics_detail.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ GTSRB 详细指标图")
    
    # 图4: CNTSSS 详细指标
    fig, ax = plt.subplots(figsize=(10, 7))
    
    cntsss_baseline_values = [
        results['CNTSSS_Baseline']['mAP@0.5'] * 100,
        results['CNTSSS_Baseline']['mAP@0.5:0.95'] * 100,
        results['CNTSSS_Baseline']['Precision'] * 100,
        results['CNTSSS_Baseline']['Recall'] * 100
    ]
    
    bars = ax.bar(x, cntsss_baseline_values, width, label='CNTSSS Baseline', 
                  color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title('CNTSSS Baseline - Detailed Metrics\n(Real Night-Time)', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cntsss_metrics_detail.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ CNTSSS 详细指标图")
    
    # 图5: 总览对比表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # GTSRB
    ax1.bar(gtsrb_methods, gtsrb_map, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('mAP@0.5 (%)', fontsize=13, fontweight='bold')
    ax1.set_title('GTSRB (Simulated Low-Light)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 80)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=70, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    for i, (bar, val) in enumerate(zip(ax1.patches, gtsrb_map)):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # CNTSSS
    cntsss_colors = ['#3498db', '#f39c12', '#e74c3c']
    ax2.bar(cntsss_methods, cntsss_map, color=cntsss_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('mAP@0.5 (%)', fontsize=13, fontweight='bold')
    ax2.set_title('CNTSSS (Real Night-Time)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 80)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=67.9, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    for i, (bar, val) in enumerate(zip(ax2.patches, cntsss_map)):
        ax2.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Comprehensive Comparison: Two Datasets', fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 综合对比图")

def generate_summary_table(results, output_dir):
    """生成结果汇总表"""
    
    # 创建汇总数据
    summary_data = []
    
    # GTSRB
    for method in ['Baseline', 'Traditional', 'Mild', 'EnlightenGAN']:
        key = f'GTSRB_{method}'
        if key in results:
            summary_data.append({
                'Dataset': 'GTSRB',
                'Method': method,
                'mAP@0.5': f"{results[key]['mAP@0.5']*100:.1f}%",
                'mAP@0.5:0.95': f"{results[key]['mAP@0.5:0.95']*100:.1f}%",
                'Precision': f"{results[key]['Precision']*100:.1f}%",
                'Recall': f"{results[key]['Recall']*100:.1f}%"
            })
    
    # CNTSSS
    for method in ['Baseline', 'Mild', 'EnlightenGAN']:
        key = f'CNTSSS_{method}'
        if key in results:
            summary_data.append({
                'Dataset': 'CNTSSS',
                'Method': method,
                'mAP@0.5': f"{results[key]['mAP@0.5']*100:.1f}%",
                'mAP@0.5:0.95': f"{results[key]['mAP@0.5:0.95']*100:.1f}%",
                'Precision': f"{results[key]['Precision']*100:.1f}%",
                'Recall': f"{results[key]['Recall']*100:.1f}%"
            })
    
    # 保存为 CSV
    df = pd.DataFrame(summary_data)
    csv_path = output_dir / 'all_results_summary.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 打印表格
    print("\n" + "="*80)
    print("  实验结果汇总表")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    print("\n✓ 已保存到:", csv_path)
    
    return df

def generate_key_findings(results, output_dir):
    """生成关键发现文本"""
    
    findings = """
# 🎯 关键实验发现（PPT 用）

## 一、核心结论

### 1. Baseline 性能已经很好
- **GTSRB (模拟低光照)**: 70.4%
- **CNTSSS (真实夜间)**: 67.9%
- **说明**: YOLO 模型已能适应低光照/夜间场景

### 2. 传统增强方法效果中性
- **GTSRB**: 70.1-70.2% (与 Baseline 持平)
- **CNTSSS**: 67.8% (与 Baseline 持平)
- **结论**: 传统增强对交通标志检测无明显帮助

### 3. EnlightenGAN 完全失败
- **GTSRB**: 39.7% (-30.7%)
- **CNTSSS**: 42.5% (-25.4%)
- **原因**: Resize 损失细节 + GAN 不适合交通标志

---

## 二、数据集对比

### GTSRB (模拟低光照场景)
- **数据量**: 31,368 张训练图
- **类别**: 43 类细分交通标志
- **场景**: Gamma 变换模拟低光照
- **Baseline mAP**: 70.4%

### CNTSSS (真实夜间场景)
- **数据量**: 3,276 张训练图
- **类别**: 3 类大类（禁止/指令/警告）
- **场景**: 真实夜间拍摄
- **Baseline mAP**: 67.9%

**一致性**: 两个数据集得出相同结论

---

## 三、方法分析

### Baseline (无增强)
**优点**:
- ✅ 简单直接
- ✅ 推理速度快（无增强开销）
- ✅ 性能已经很好（67-70%）

**结论**: 对于交通标志检测，直接训练原始数据即可

### 传统方法 (CLAHE + Gamma)
**优点**:
- ✅ 速度快
- ✅ 参数可调

**缺点**:
- ⚠️ 效果与 Baseline 持平（±0.2%）
- ⚠️ 无明显收益

**结论**: 性价比低，不推荐使用

### EnlightenGAN (深度学习)
**优点**:
- ✅ 理论上应该更智能

**缺点**:
- ❌ Resize 损失细节（256x256）
- ❌ 不适合交通标志任务
- ❌ 性能大幅下降（-25-30%）

**结论**: 不适用于该任务

---

## 四、实用建议

### 对于夜间交通标志检测：

✅ **推荐方案**: 直接使用 Baseline
- 无需图像增强
- 训练原始夜间/低光照数据
- 部署时直接检测，速度快

❌ **不推荐**: 复杂的增强流水线
- 计算开销大
- 性能无提升甚至下降
- 维护成本高

---

## 五、学术价值

### 负面结果的价值
- ✅ 证明了"不是所有任务都需要增强"
- ✅ 为低光照检测研究提供参考
- ✅ 说明任务特性很重要

### 实验的完整性
- ✅ 2 个数据集（模拟 + 真实）
- ✅ 4-6 组对比实验
- ✅ 系统性分析

---

## 六、PPT 重点

### Slide 重点内容

**问题提出**:
- 低光照/夜间条件影响交通标志检测
- 图像增强是否能提升性能？

**实验设计**:
- 2个数据集 × 3-4种方法
- 系统性对比

**关键发现**:
1. Baseline 已经很好 (67-70%)
2. 传统增强无明显收益 (±0.2%)
3. EnlightenGAN 不适用 (-25-30%)

**最终建议**:
- 直接训练原始数据
- 无需复杂增强流水线
- 简单有效的方案最好

---

## 七、数据表格（可直接用于PPT）

"""
    
    # 添加数据表格
    findings += "\n### GTSRB 实验结果\n\n"
    findings += "| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |\n"
    findings += "|--------|---------|--------------|-----------|--------|\n"
    
    for method in ['Baseline', 'Traditional', 'Mild', 'EnlightenGAN']:
        key = f'GTSRB_{method}'
        if key in results:
            r = results[key]
            findings += f"| {method} | {r['mAP@0.5']*100:.1f}% | {r['mAP@0.5:0.95']*100:.1f}% | {r['Precision']*100:.1f}% | {r['Recall']*100:.1f}% |\n"
    
    findings += "\n### CNTSSS 实验结果\n\n"
    findings += "| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |\n"
    findings += "|--------|---------|--------------|-----------|--------|\n"
    
    for method in ['Baseline', 'Mild', 'EnlightenGAN']:
        key = f'CNTSSS_{method}'
        if key in results:
            r = results[key]
            findings += f"| {method} | {r['mAP@0.5']*100:.1f}% | {r['mAP@0.5:0.95']*100:.1f}% | {r['Precision']*100:.1f}% | {r['Recall']*100:.1f}% |\n"
    
    # 保存
    findings_path = output_dir / 'KEY_FINDINGS_FOR_PPT.txt'
    with open(findings_path, 'w', encoding='utf-8') as f:
        f.write(findings)
    
    print(f"\n✓ 关键发现已保存: {findings_path}")

def main():
    print("="*70)
    print("  生成 PPT 素材")
    print("="*70)
    
    output_dir = Path("ppt_materials")
    output_dir.mkdir(exist_ok=True)
    
    # 加载结果
    print("\n📊 加载实验结果...")
    results = load_experiment_results()
    
    print(f"✅ 已加载 {len(results)} 组实验结果\n")
    
    # 生成图表
    print("📈 生成对比图表...")
    generate_comparison_charts(results, output_dir)
    
    # 生成汇总表
    print("\n📋 生成结果汇总表...")
    df = generate_summary_table(results, output_dir)
    
    # 生成关键发现
    print("\n📝 生成关键发现...")
    generate_key_findings(results, output_dir)
    
    # 复制重要图片
    print("\n📁 复制关键训练结果图...")
    import shutil
    
    key_images = [
        ('experiments/exp1_baseline/run8/results.png', 'gtsrb_baseline_training_curve.png'),
        ('experiments/exp1_baseline/run8/confusion_matrix.png', 'gtsrb_baseline_confusion_matrix.png'),
        ('experiments/cntsss_baseline/run/results.png', 'cntsss_baseline_training_curve.png'),
        ('experiments/cntsss_baseline/run/confusion_matrix.png', 'cntsss_baseline_confusion_matrix.png'),
    ]
    
    copied = 0
    for src, dst in key_images:
        src_path = Path(src)
        if src_path.exists():
            shutil.copy2(src_path, output_dir / dst)
            print(f"  ✓ {dst}")
            copied += 1
    
    # 最终汇总
    print("\n" + "="*70)
    print("  ✅ PPT 素材生成完成！")
    print("="*70)
    
    print(f"\n📂 所有素材保存在: {output_dir.absolute()}\n")
    print("生成的文件：")
    print("  📊 图表:")
    print("     • gtsrb_comparison.png - GTSRB 方法对比")
    print("     • cntsss_comparison.png - CNTSSS 方法对比")
    print("     • comprehensive_comparison.png - 综合对比")
    print("     • gtsrb_metrics_detail.png - GTSRB 详细指标")
    print("     • cntsss_metrics_detail.png - CNTSSS 详细指标")
    
    print("\n  📋 数据:")
    print("     • all_results_summary.csv - 所有结果汇总表")
    print("     • KEY_FINDINGS_FOR_PPT.txt - 关键发现和结论")
    
    print("\n  🖼️  训练图:")
    print(f"     • {copied} 张训练曲线和混淆矩阵")
    
    print("\n" + "="*70)
    print("  💡 使用建议")
    print("="*70)
    print("\n1. 打开 ppt_materials/ 文件夹查看所有图表")
    print("2. 阅读 KEY_FINDINGS_FOR_PPT.txt 获取讲解要点")
    print("3. 使用图表制作 PPT")
    print("4. 参考汇总表制作对比 Slide")
    print("\n祝 PPT 制作顺利！💪\n")

if __name__ == '__main__':
    main()

