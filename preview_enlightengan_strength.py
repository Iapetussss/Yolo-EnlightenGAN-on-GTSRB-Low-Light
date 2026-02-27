#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预览不同强度的EnlightenGAN增强效果
随机选几张图，对比不同混合比例
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import onnxruntime as ort
import random

def preprocess_image(image, target_size=256):
    """预处理图像用于模型输入"""
    image = cv2.resize(image, (target_size, target_size))
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = image.astype(np.float32) / 127.5 - 1.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    
    return image

def postprocess_image(output):
    """后处理模型输出"""
    output = np.squeeze(output, axis=0)
    output = np.transpose(output, (1, 2, 0))
    output = (output + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def blend_images(original, enhanced, alpha):
    """混合原图和增强图"""
    original_resized = cv2.resize(original, (256, 256))
    blended = cv2.addWeighted(enhanced, alpha, original_resized, 1-alpha, 0)
    return blended.astype(np.uint8)

def main():
    print("="*70)
    print("  预览不同强度的EnlightenGAN增强效果")
    print("="*70)
    
    # 路径配置
    model_path = Path("models/enlightengan/enlightengan.onnx")
    test_dir = Path("data/baseline_lowlight_dataset/images/train")
    output_dir = Path("results/enlightengan_strength_preview")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return
    
    if not test_dir.exists():
        print(f"❌ 测试目录不存在: {test_dir}")
        return
    
    # 加载模型
    print(f"\n📥 加载 EnlightenGAN 模型...")
    try:
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        print(f"✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 随机选择5张图像
    all_images = sorted(test_dir.glob("*.png"))
    if len(all_images) == 0:
        print("❌ 没有找到图像")
        return
    
    num_images = min(5, len(all_images))
    selected_images = random.sample(all_images, num_images)
    
    print(f"🎲 随机选择 {num_images} 张图像")
    print(f"📂 输出目录: {output_dir}\n")
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 不同强度
    alphas = [0.5, 0.6, 0.7, 0.8, 1.0]
    alpha_labels = ['50%', '60%', '70%', '80%', '100%']
    
    # 处理每张图像
    for idx, img_path in enumerate(selected_images, 1):
        print(f"[{idx}/{num_images}] 处理 {img_path.name}...", end=" ")
        
        try:
            # 读取原图
            original = cv2.imread(str(img_path))
            if original is None:
                print("❌")
                continue
            
            # EnlightenGAN 增强
            input_data = preprocess_image(original)
            output = session.run(None, {input_name: input_data})
            enhanced_full = postprocess_image(output[0])
            
            # 创建对比图（1行6列：原图 + 5种强度）
            fig, axes = plt.subplots(1, 6, figsize=(18, 3))
            
            # 原图
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            axes[0].imshow(cv2.resize(original_rgb, (256, 256)))
            axes[0].set_title('Original\n(Low-light)', fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            brightness_orig = np.mean(original)
            axes[0].text(0.5, -0.15, f'亮度: {brightness_orig:.1f}', 
                        transform=axes[0].transAxes, ha='center', fontsize=9)
            
            # 不同强度的增强
            for i, (alpha, label) in enumerate(zip(alphas, alpha_labels)):
                if alpha == 1.0:
                    # 完全增强
                    blended = enhanced_full
                    title_color = 'red'
                    title = f'{label}\n(Current)'
                else:
                    # 混合
                    blended = blend_images(original, enhanced_full, alpha)
                    if alpha == 0.7:
                        title_color = 'green'
                        title = f'{label}\n⭐ 推荐'
                    else:
                        title_color = 'black'
                        title = f'{label}'
                
                blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                axes[i+1].imshow(blended_rgb)
                axes[i+1].set_title(title, fontsize=10, fontweight='bold', color=title_color)
                axes[i+1].axis('off')
                
                brightness = np.mean(blended)
                axes[i+1].text(0.5, -0.15, f'亮度: {brightness:.1f}', 
                             transform=axes[i+1].transAxes, ha='center', fontsize=9)
            
            plt.suptitle(f'EnlightenGAN 强度对比 - {img_path.stem}', 
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # 保存
            output_path = output_dir / f"strength_preview_{idx}_{img_path.stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓")
            
        except Exception as e:
            print(f"❌ {e}")
            continue
    
    # 生成说明
    print(f"\n{'='*70}")
    print(f"✅ 预览图生成完成！")
    print(f"{'='*70}\n")
    print(f"📂 结果保存在: {output_dir}/\n")
    print(f"强度说明：")
    print(f"  • 50%: 最温和（保留50%原图）")
    print(f"  • 60%: 温和")
    print(f"  • 70%: 轻度 ⭐ 推荐（平衡）")
    print(f"  • 80%: 中等")
    print(f"  • 100%: 完全增强（当前39.7%结果）")
    print(f"\n💡 建议：")
    print(f"  1. 查看生成的对比图")
    print(f"  2. 选择看起来最好的强度")
    print(f"  3. 用该强度批量增强数据集")
    print()

if __name__ == '__main__':
    main()

