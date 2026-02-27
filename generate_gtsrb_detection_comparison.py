#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成GTSRB数据集的检测结果对比图
专门展示EnlightenGAN对小目标交通标志检测的不利影响
(a) 原始低光照图像 + Baseline检测结果
(b) EnlightenGAN增强后图像 + 检测结果（展示性能下降）
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_detections(image, results, conf_threshold=0.25, color=(0, 255, 0)):
    """在图像上绘制检测结果"""
    img_with_detections = image.copy()
    
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            # 获取检测框信息
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
                
            # 获取坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制检测框
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f'{conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_with_detections, (x1, y1 - label_size[1] - 5), 
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_with_detections, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img_with_detections

def main():
    print("="*70)
    print("  生成GTSRB检测结果对比图（展示EnlightenGAN的不利影响）")
    print("="*70)
    
    # 路径配置 - GTSRB数据集
    original_dir = Path("data/baseline_lowlight_dataset/images/test")
    enlightengan_dir = Path("data/enhanced/images/test")
    
    # 模型路径
    baseline_model_path = Path("experiments/exp1_baseline/run8/weights/best.pt")
    enlightengan_model_path = Path("experiments/exp3_enlightengan/run/weights/best.pt")
    
    output_dir = Path("results/gtsrb_detection_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查目录
    if not original_dir.exists():
        print(f"❌ 原始图像目录不存在: {original_dir}")
        return
    
    if not enlightengan_dir.exists():
        print(f"❌ EnlightenGAN增强图像目录不存在: {enlightengan_dir}")
        return
    
    # 检查模型文件
    if not baseline_model_path.exists():
        print(f"❌ Baseline模型不存在: {baseline_model_path}")
        return
    
    if not enlightengan_model_path.exists():
        print(f"❌ EnlightenGAN模型不存在: {enlightengan_model_path}")
        return
    
    # 加载模型
    print("\n📦 加载模型...")
    try:
        baseline_model = YOLO(str(baseline_model_path))
        print(f"  ✓ Baseline模型加载成功 (mAP@0.5: 70.40%)")
    except Exception as e:
        print(f"  ✗ Baseline模型加载失败: {e}")
        return
    
    try:
        enlightengan_model = YOLO(str(enlightengan_model_path))
        print(f"  ✓ EnlightenGAN模型加载成功 (mAP@0.5: 39.66%)")
    except Exception as e:
        print(f"  ✗ EnlightenGAN模型加载失败: {e}")
        return
    
    # 获取所有图像
    all_images = sorted(original_dir.glob("*.png")) + sorted(original_dir.glob("*.jpg"))
    
    if len(all_images) == 0:
        print("❌ 没有找到图像")
        return
    
    # 随机选择几张图像（建议3-5张）
    num_images = min(5, len(all_images))
    selected_images = random.sample(all_images, num_images)
    
    print(f"\n🎲 随机选择 {num_images} 张图像生成对比图")
    print(f"📂 输出目录: {output_dir}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, img_path in enumerate(selected_images, 1):
        print(f"[{i}/{num_images}] 处理 {img_path.name}...", end=" ")
        
        try:
            # 读取原始图像
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                print("❌ 原始图像读取失败")
                fail_count += 1
                continue
            
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # 读取EnlightenGAN增强图像
            gan_path = enlightengan_dir / img_path.name
            if not gan_path.exists():
                print(f"❌ EnlightenGAN增强图像不存在: {gan_path}")
                fail_count += 1
                continue
            
            enlightengan_img = cv2.imread(str(gan_path))
            if enlightengan_img is None:
                print("❌ EnlightenGAN增强图像读取失败")
                fail_count += 1
                continue
            
            enlightengan_rgb = cv2.cvtColor(enlightengan_img, cv2.COLOR_BGR2RGB)
            
            # 使用Baseline模型检测原始图像
            baseline_results = baseline_model(original_img, verbose=False)
            original_with_det = draw_detections(original_rgb, baseline_results, color=(0, 255, 0))
            
            # 使用EnlightenGAN模型检测增强图像
            enlightengan_results = enlightengan_model(enlightengan_img, verbose=False)
            enlightengan_with_det = draw_detections(enlightengan_rgb, enlightengan_results, color=(255, 0, 0))
            
            # 计算检测数量
            baseline_det_count = len(baseline_results[0].boxes) if baseline_results and len(baseline_results) > 0 and baseline_results[0].boxes is not None else 0
            enlightengan_det_count = len(enlightengan_results[0].boxes) if enlightengan_results and len(enlightengan_results) > 0 and enlightengan_results[0].boxes is not None else 0
            
            # 创建对比图（1行2列）
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # (a) 原始低光照图像 + Baseline检测结果
            axes[0].imshow(original_with_det)
            axes[0].set_title('(a) 原始低光照图像 + Baseline检测结果\n(mAP@0.5: 70.40%)', 
                            fontsize=14, fontweight='bold', pad=10)
            axes[0].axis('off')
            axes[0].text(0.5, -0.05, f'检测数量: {baseline_det_count}', 
                       transform=axes[0].transAxes, ha='center', fontsize=12, color='green')
            
            # (b) EnlightenGAN增强后图像 + 检测结果
            axes[1].imshow(enlightengan_with_det)
            axes[1].set_title('(b) EnlightenGAN增强后图像 + 检测结果\n(mAP@0.5: 39.66% ⚠️)', 
                            fontsize=14, fontweight='bold', color='red', pad=10)
            axes[1].axis('off')
            axes[1].text(0.5, -0.05, f'检测数量: {enlightengan_det_count} (下降 {baseline_det_count - enlightengan_det_count})', 
                       transform=axes[1].transAxes, ha='center', fontsize=12, color='red')
            
            # 添加说明文字
            fig.text(0.5, 0.02, 
                    f'EnlightenGAN增强导致检测性能显著下降：mAP从70.40%降至39.66%\n'
                    f'原因：resize到256×256导致小目标交通标志细节丢失',
                    ha='center', fontsize=11, style='italic', color='red')
            
            plt.suptitle(f'GTSRB Detection Comparison - {img_path.stem}\n'
                        f'EnlightenGAN对小目标检测的不利影响', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)
            
            # 保存图像
            output_path = output_dir / f"gtsrb_detection_comparison_{i:02d}_{img_path.stem}.png"
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 已保存 (Baseline: {baseline_det_count}, EnlightenGAN: {enlightengan_det_count})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue
    
    # 统计
    print(f"\n{'='*70}")
    print(f"✅ 对比图生成完成！")
    print(f"{'='*70}\n")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"\n📂 结果保存在: {output_dir}/")
    print(f"\n💡 这些对比图展示了EnlightenGAN对小目标检测的不利影响：")
    print(f"  • Baseline mAP@0.5: 70.40%")
    print(f"  • EnlightenGAN mAP@0.5: 39.66% (下降30.74%)")
    print(f"  • 原因：resize到256×256导致小目标细节丢失")
    print()

if __name__ == '__main__':
    main()

