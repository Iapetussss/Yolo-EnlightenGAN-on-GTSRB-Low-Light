#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从results.csv生成训练曲线图（results.png）
用于CNTSSS EnlightenGAN实验
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def smooth(y, f=0.6):
    """平滑曲线"""
    n = len(y)
    smoothed = np.zeros(n)
    for i in range(n):
        if i == 0:
            smoothed[i] = y[i]
        else:
            smoothed[i] = f * smoothed[i-1] + (1-f) * y[i]
    return smoothed

def generate_results_plot(csv_path, output_path):
    """生成训练曲线图"""
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 创建2x5的子图
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Training Results', fontsize=16, fontweight='bold', y=0.995)
    
    epochs = df['epoch'].values
    
    # 第一行：训练指标
    # 1. train/box_loss
    ax = axes[0, 0]
    y = df['train/box_loss'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('train/box_loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. train/cls_loss
    ax = axes[0, 1]
    y = df['train/cls_loss'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('train/cls_loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. train/dfl_loss
    ax = axes[0, 2]
    y = df['train/dfl_loss'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('train/dfl_loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. metrics/precision(B)
    ax = axes[0, 3]
    y = df['metrics/precision(B)'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('metrics/precision(B)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. metrics/recall(B)
    ax = axes[0, 4]
    y = df['metrics/recall(B)'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('metrics/recall(B)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 第二行：验证指标
    # 1. val/box_loss
    ax = axes[1, 0]
    y = df['val/box_loss'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('val/box_loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. val/cls_loss
    ax = axes[1, 1]
    y = df['val/cls_loss'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('val/cls_loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. val/dfl_loss
    ax = axes[1, 2]
    y = df['val/dfl_loss'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('val/dfl_loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. metrics/mAP50(B)
    ax = axes[1, 3]
    y = df['metrics/mAP50(B)'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('metrics/mAP50(B)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. metrics/mAP50-95(B)
    ax = axes[1, 4]
    y = df['metrics/mAP50-95(B)'].values
    ax.plot(epochs, y, 'b-', label='results', linewidth=1.5, markersize=3)
    ax.plot(epochs, smooth(y), 'orange', linestyle='--', label='smooth', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5:0.95')
    ax.set_title('metrics/mAP50-95(B)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线图已保存: {output_path}")
    plt.close()

def main():
    # CNTSSS EnlightenGAN实验
    csv_path = Path("experiments/cntsss_enlightengan_mild_enhanced/run/results.csv")
    output_path = Path("experiments/cntsss_enlightengan_mild_enhanced/run/results.png")
    
    if not csv_path.exists():
        print(f"❌ CSV文件不存在: {csv_path}")
        return
    
    print(f"从 {csv_path} 生成训练曲线图...")
    generate_results_plot(csv_path, output_path)
    print(f"完成！")

if __name__ == '__main__':
    main()

