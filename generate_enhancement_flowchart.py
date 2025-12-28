#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成传统低光照图像增强流程示意图
流程：Low-light Image → Multi-Scale Retinex → CLAHE (LAB space) → Gamma Correction → Enhanced Image
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_flowchart():
    """创建流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    # 设置白底
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 定义步骤
    steps = [
        ("Low-light\nImage", 1.0),
        ("Multi-Scale\nRetinex", 3.0),
        ("CLAHE\n(LAB space)", 5.5),
        ("Gamma\nCorrection", 7.5),
        ("Enhanced\nImage", 9.5)
    ]
    
    # 绘制步骤框
    boxes = []
    for i, (text, x) in enumerate(steps):
        # 创建圆角矩形框
        box = FancyBboxPatch(
            (x - 0.6, 0.6), 1.2, 0.8,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor='white',
            linewidth=2
        )
        ax.add_patch(box)
        boxes.append(box)
        
        # 添加文字
        ax.text(x, 1.0, text, 
               ha='center', va='center',
               fontsize=11, fontweight='bold',
               color='black',
               family='sans-serif')
    
    # 绘制箭头
    arrow_props = dict(
        arrowstyle='->',
        lw=2,
        color='black'
    )
    
    # 箭头位置
    arrow_positions = [
        (1.6, 1.0, 2.4, 1.0),  # Low-light → Retinex
        (3.6, 1.0, 4.9, 1.0),  # Retinex → CLAHE
        (6.1, 1.0, 6.9, 1.0),  # CLAHE → Gamma
        (8.1, 1.0, 8.9, 1.0),  # Gamma → Enhanced
    ]
    
    for x1, y1, x2, y2 in arrow_positions:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            **arrow_props
        )
        ax.add_patch(arrow)
    
    # 添加标题
    ax.text(5.0, 1.8, 'Traditional Low-light Image Enhancement Pipeline',
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            color='black',
            family='sans-serif')
    
    # 添加细节说明（可选，放在下方）
    details = [
        "Scales: [15, 80, 250]",
        "clipLimit=2.5",
        "γ = 1.2"
    ]
    
    detail_x = [3.0, 5.5, 7.5]
    for i, (x, detail) in enumerate(zip(detail_x, details)):
        ax.text(x, 0.2, detail,
               ha='center', va='center',
               fontsize=9,
               color='black',
               style='italic',
               family='sans-serif')
    
    plt.tight_layout()
    
    # 保存
    output_path = "results/enhancement_flowchart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"流程图已保存: {output_path}")
    
    # 也保存为PDF格式（适合报告）
    output_path_pdf = "results/enhancement_flowchart.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"流程图已保存: {output_path_pdf}")
    
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs("results", exist_ok=True)
    create_flowchart()

