#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成三种方法的对比图
对比：原始低光照 vs 传统增强 vs EnlightenGAN增强
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

def main():
    print("="*70)
    print("  生成增强方法对比图")
    print("="*70)
    
    # 路径配置
    lowlight_dir = Path("data/baseline_lowlight_dataset/images/train")
    traditional_dir = Path("data/traditional_enhanced/images/train")
    enlightengan_dir = Path("data/enhanced/images/train")
    output_dir = Path("results/enhancement_comparison")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查目录
    if not lowlight_dir.exists():
        print(f"❌ 低光照目录不存在: {lowlight_dir}")
        return
    
    if not traditional_dir.exists():
        print(f"❌ 传统增强目录不存在: {traditional_dir}")
        return
    
    if not enlightengan_dir.exists():
        print(f"❌ EnlightenGAN目录不存在: {enlightengan_dir}")
        return
    
    # 获取所有图像
    all_images = sorted(lowlight_dir.glob("*.png"))
    
    if len(all_images) == 0:
        print("❌ 没有找到图像")
        return
    
    # 随机选择10张图像
    num_images = min(10, len(all_images))
    selected_images = random.sample(all_images, num_images)
    
    print(f"\n🎲 随机选择 {num_images} 张图像生成对比图")
    print(f"📂 输出目录: {output_dir}\n")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    success_count = 0
    fail_count = 0
    
    for i, img_path in enumerate(selected_images, 1):
        print(f"[{i}/{num_images}] 处理 {img_path.name}...", end=" ")
        
        try:
            # 读取三张图像
            lowlight_img = cv2.imread(str(img_path))
            traditional_img = cv2.imread(str(traditional_dir / img_path.name))
            enlightengan_img = cv2.imread(str(enlightengan_dir / img_path.name))
            
            if lowlight_img is None or traditional_img is None or enlightengan_img is None:
                print("❌ 图像读取失败")
                fail_count += 1
                continue
            
            # 转换为RGB
            lowlight_rgb = cv2.cvtColor(lowlight_img, cv2.COLOR_BGR2RGB)
            traditional_rgb = cv2.cvtColor(traditional_img, cv2.COLOR_BGR2RGB)
            enlightengan_rgb = cv2.cvtColor(enlightengan_img, cv2.COLOR_BGR2RGB)
            
            # 创建对比图（1行3列）
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始低光照
            axes[0].imshow(lowlight_rgb)
            axes[0].set_title('Low-light (Baseline)', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # 计算平均亮度
            brightness_low = np.mean(lowlight_rgb)
            axes[0].text(0.5, -0.1, f'Brightness: {brightness_low:.1f}', 
                        transform=axes[0].transAxes, ha='center', fontsize=11)
            
            # 传统增强
            axes[1].imshow(traditional_rgb)
            axes[1].set_title('Traditional Enhancement', fontsize=14, fontweight='bold', color='green')
            axes[1].axis('off')
            
            brightness_trad = np.mean(traditional_rgb)
            axes[1].text(0.5, -0.1, f'Brightness: {brightness_trad:.1f} (+{brightness_trad-brightness_low:.1f})', 
                        transform=axes[1].transAxes, ha='center', fontsize=11, color='green')
            
            # EnlightenGAN增强
            axes[2].imshow(enlightengan_rgb)
            axes[2].set_title('EnlightenGAN Enhancement', fontsize=14, fontweight='bold', color='blue')
            axes[2].axis('off')
            
            brightness_gan = np.mean(enlightengan_rgb)
            axes[2].text(0.5, -0.1, f'Brightness: {brightness_gan:.1f} (+{brightness_gan-brightness_low:.1f})', 
                        transform=axes[2].transAxes, ha='center', fontsize=11, color='blue')
            
            plt.suptitle(f'Enhancement Methods Comparison - {img_path.stem}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # 保存图像
            output_path = output_dir / f"comparison_{i:02d}_{img_path.stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 已保存")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            fail_count += 1
            continue
    
    # 生成总结对比图（多图展示）
    print(f"\n生成总结对比图...")
    generate_summary_comparison(selected_images[:4], lowlight_dir, traditional_dir, 
                                enlightengan_dir, output_dir)
    
    # 统计
    print(f"\n{'='*70}")
    print(f"✅ 对比图生成完成！")
    print(f"{'='*70}\n")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"\n📂 结果保存在: {output_dir}/")
    print(f"\n可用于PPT的对比图：")
    for i in range(1, min(success_count + 1, 6)):
        print(f"  • comparison_{i:02d}_*.png")
    print(f"  • summary_comparison.png (多图总览)")
    print()

def generate_summary_comparison(image_paths, lowlight_dir, traditional_dir, 
                                enlightengan_dir, output_dir):
    """生成4张图像的总结对比"""
    try:
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        for i, img_path in enumerate(image_paths):
            # 读取图像
            lowlight_img = cv2.imread(str(img_path))
            traditional_img = cv2.imread(str(traditional_dir / img_path.name))
            enlightengan_img = cv2.imread(str(enlightengan_dir / img_path.name))
            
            if lowlight_img is None or traditional_img is None or enlightengan_img is None:
                continue
            
            # 转换为RGB
            lowlight_rgb = cv2.cvtColor(lowlight_img, cv2.COLOR_BGR2RGB)
            traditional_rgb = cv2.cvtColor(traditional_img, cv2.COLOR_BGR2RGB)
            enlightengan_rgb = cv2.cvtColor(enlightengan_img, cv2.COLOR_BGR2RGB)
            
            # 显示三张图
            axes[i, 0].imshow(lowlight_rgb)
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('Low-light', fontsize=12, fontweight='bold')
            
            axes[i, 1].imshow(traditional_rgb)
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title('Traditional', fontsize=12, fontweight='bold', color='green')
            
            axes[i, 2].imshow(enlightengan_rgb)
            axes[i, 2].axis('off')
            if i == 0:
                axes[i, 2].set_title('EnlightenGAN', fontsize=12, fontweight='bold', color='blue')
        
        plt.suptitle('Enhancement Methods Comparison - Summary', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / "summary_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 总结对比图已保存")
        
    except Exception as e:
        print(f"  ❌ 总结对比图生成失败: {e}")

if __name__ == '__main__':
    main()

