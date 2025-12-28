#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预览温和增强效果
对比：原始低光照 vs 温和增强
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

def enhance_mild(image):
    """温和增强（推荐参数）"""
    # 1. 轻度 Gamma 校正
    gamma = 1.15
    enhanced = np.power(image.astype(np.float32) / 255.0, 1/gamma)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    # 2. 轻度 CLAHE
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def enhance_strong(image):
    """当前传统方法（已训练的参数）"""
    # Multi-Scale Retinex
    img_float = image.astype(np.float32) / 255.0
    scales = [15, 80, 250]
    retinex = np.zeros_like(img_float)
    
    for scale in scales:
        blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
        blurred = np.maximum(blurred, 0.001)
        retinex += np.log10(img_float + 1) - np.log10(blurred + 1)
    
    retinex = retinex / len(scales)
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-8)
    
    # CLAHE
    lab = cv2.cvtColor((retinex * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Gamma
    gamma = 1.2
    enhanced = np.power(enhanced.astype(np.float32) / 255.0, gamma)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    return enhanced

def main():
    print("="*70)
    print("  预览温和增强 vs 当前传统增强")
    print("="*70)
    
    # 路径配置
    lowlight_dir = Path("data/baseline_lowlight_dataset/images/train")
    output_dir = Path("results/mild_enhancement_preview")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查
    if not lowlight_dir.exists():
        print(f"❌ 低光照目录不存在: {lowlight_dir}")
        return
    
    # 随机选择10张图像
    all_images = sorted(lowlight_dir.glob("*.png"))
    if len(all_images) == 0:
        print("❌ 没有找到图像")
        return
    
    num_images = min(10, len(all_images))
    selected_images = random.sample(all_images, num_images)
    
    print(f"\n🎲 随机选择 {num_images} 张图像")
    print(f"📂 输出目录: {output_dir}\n")
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    success_count = 0
    
    # 处理每张图像
    for i, img_path in enumerate(selected_images, 1):
        print(f"[{i}/{num_images}] 处理 {img_path.name}...", end=" ")
        
        try:
            # 读取原图
            original = cv2.imread(str(img_path))
            if original is None:
                print("❌")
                continue
            
            # 温和增强
            mild_enhanced = enhance_mild(original)
            
            # 当前传统增强
            strong_enhanced = enhance_strong(original)
            
            # 创建对比图（1行3列）
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始低光照
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            axes[0].imshow(original_rgb)
            axes[0].set_title('Low-light (Baseline)', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            brightness_orig = np.mean(original_rgb)
            axes[0].text(0.5, -0.1, f'亮度: {brightness_orig:.1f}', 
                        transform=axes[0].transAxes, ha='center', fontsize=11)
            
            # 温和增强
            mild_rgb = cv2.cvtColor(mild_enhanced, cv2.COLOR_BGR2RGB)
            axes[1].imshow(mild_rgb)
            axes[1].set_title('Mild Enhancement ⭐\n(Gamma 1.15 + CLAHE 1.5)', 
                            fontsize=14, fontweight='bold', color='green')
            axes[1].axis('off')
            
            brightness_mild = np.mean(mild_rgb)
            axes[1].text(0.5, -0.1, f'亮度: {brightness_mild:.1f} (+{brightness_mild-brightness_orig:.1f})', 
                        transform=axes[1].transAxes, ha='center', fontsize=11, color='green')
            
            # 当前传统增强
            strong_rgb = cv2.cvtColor(strong_enhanced, cv2.COLOR_BGR2RGB)
            axes[2].imshow(strong_rgb)
            axes[2].set_title('Current Traditional\n(Retinex + CLAHE 2.5) - mAP 70.1%', 
                            fontsize=14, fontweight='bold', color='blue')
            axes[2].axis('off')
            
            brightness_strong = np.mean(strong_rgb)
            axes[2].text(0.5, -0.1, f'亮度: {brightness_strong:.1f} (+{brightness_strong-brightness_orig:.1f})', 
                        transform=axes[2].transAxes, ha='center', fontsize=11, color='blue')
            
            plt.suptitle(f'Enhancement Methods Comparison - {img_path.stem}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # 保存
            output_path = output_dir / f"mild_comparison_{i:02d}_{img_path.stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {e}")
            continue
    
    # 统计
    print(f"\n{'='*70}")
    print(f"✅ 对比图生成完成！")
    print(f"{'='*70}\n")
    print(f"成功: {success_count} 张")
    print(f"\n📂 结果保存在: {output_dir}/")
    print(f"\n对比说明：")
    print(f"  • 左：原始低光照 (Baseline mAP 70.4%)")
    print(f"  • 中：温和增强 ⭐ (更轻度，可能更好)")
    print(f"  • 右：当前传统增强 (mAP 70.1%)")
    print(f"\n💡 观察哪个增强效果更好：")
    print(f"  • 细节是否保留？")
    print(f"  • 颜色是否自然？")
    print(f"  • 是否过度增强？")
    print()

if __name__ == '__main__':
    main()

