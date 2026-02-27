#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成包含检测结果的对比图
(a) 原始低光照图像 + Baseline 检测结果
(b) 传统增强后图像 + 检测结果
(c) EnlightenGAN 增强后图像 + 检测结果
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random

def traditional_enhance(image):
    """传统增强方法（Multi-Scale Retinex + CLAHE + Gamma）"""
    # 转换为浮点数
    img_float = image.astype(np.float32) / 255.0
    
    # 1. Multi-Scale Retinex
    scales = [15, 80, 250]
    retinex = np.zeros_like(img_float)
    
    for scale in scales:
        # 对每个通道应用高斯模糊
        blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
        # 防止除零
        blurred = np.maximum(blurred, 0.001)
        # 计算Retinex
        retinex += np.log10(img_float + 1) - np.log10(blurred + 1)
    
    retinex = retinex / len(scales)
    
    # 归一化到[0, 1]
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-8)
    
    # 2. CLAHE增强
    lab = cv2.cvtColor((retinex * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. Gamma校正
    gamma = 1.2
    enhanced = np.power(enhanced.astype(np.float32) / 255.0, gamma)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    return enhanced

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_detections(image, results, conf_threshold=0.25):
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
            
            # 获取类别
            cls = int(box.cls[0])
            
            # 绘制检测框（绿色，线宽2）
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f'{conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_with_detections, (x1, y1 - label_size[1] - 5), 
                        (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img_with_detections, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img_with_detections

def main():
    print("="*70)
    print("  生成包含检测结果的对比图")
    print("="*70)
    
    # 路径配置
    original_dir = Path("data/CNTSSS/test/images")
    
    # EnlightenGAN增强图像路径（已确认存在）
    enlightengan_dir = Path("data/CNTSSS_enlightengan_mild/test/images")
    
    # 传统增强图像路径（尝试多个可能路径）
    possible_traditional_dirs = [
        Path("data/CNTSSS_mild_enhanced/test/images"),  # 可能是传统增强
        Path("data/traditional_enhanced/images/test"),
        Path("data/CNTSSS/traditional_enhanced/test/images"),
    ]
    
    # 模型路径
    baseline_model_path = Path("experiments/cntsss_baseline/run/weights/best.pt")
    traditional_model_path = Path("experiments/exp2_traditional/run/weights/best.pt")
    enlightengan_model_path = Path("experiments/exp3_enlightengan/run/weights/best.pt")
    
    # 如果标准路径不存在，尝试其他路径
    if not baseline_model_path.exists():
        baseline_model_path = Path("experiments/exp1_baseline/run8/weights/best.pt")
    
    output_dir = Path("results/detection_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查目录
    if not original_dir.exists():
        print(f"❌ 原始图像目录不存在: {original_dir}")
        print("   请检查路径配置")
        return
    
    # 检查模型文件
    models_to_load = {}
    if baseline_model_path.exists():
        models_to_load['baseline'] = baseline_model_path
    else:
        print(f"⚠️  Baseline模型不存在: {baseline_model_path}")
        print("   将使用通用模型进行检测")
    
    if traditional_model_path.exists():
        models_to_load['traditional'] = traditional_model_path
    else:
        print(f"⚠️  传统增强模型不存在: {traditional_model_path}")
    
    if enlightengan_model_path.exists():
        models_to_load['enlightengan'] = enlightengan_model_path
    else:
        print(f"⚠️  EnlightenGAN模型不存在: {enlightengan_model_path}")
    
    if len(models_to_load) == 0:
        print("\n❌ 没有找到任何模型文件")
        print("   请先训练模型或检查模型路径")
        return
    
    # 加载模型
    print("\n📦 加载模型...")
    models = {}
    for name, path in models_to_load.items():
        try:
            models[name] = YOLO(str(path))
            print(f"  ✓ {name} 模型加载成功")
        except Exception as e:
            print(f"  ✗ {name} 模型加载失败: {e}")
    
    if len(models) == 0:
        print("\n❌ 没有成功加载任何模型")
        return
    
    # 获取所有图像
    all_images = sorted(original_dir.glob("*.jpg"))
    
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
            
            # 读取增强后的图像（如果存在）
            traditional_img = None
            enlightengan_img = None
            
            # 尝试找到传统增强图像
            traditional_img = None
            for trad_dir in possible_traditional_dirs:
                if trad_dir.exists():
                    trad_path = trad_dir / img_path.name
                    if trad_path.exists():
                        traditional_img = cv2.imread(str(trad_path))
                        if traditional_img is not None:
                            traditional_rgb = cv2.cvtColor(traditional_img, cv2.COLOR_BGR2RGB)
                            break
            
            # 如果找不到，实时生成传统增强图像
            if traditional_img is None:
                print("(实时生成传统增强图像)", end=" ")
                traditional_img = traditional_enhance(original_img)
                traditional_rgb = cv2.cvtColor(traditional_img, cv2.COLOR_BGR2RGB)
            
            # 读取EnlightenGAN增强图像（使用确认存在的路径）
            enlightengan_img = None
            if enlightengan_dir.exists():
                gan_path = enlightengan_dir / img_path.name
                if gan_path.exists():
                    enlightengan_img = cv2.imread(str(gan_path))
                    if enlightengan_img is not None:
                        enlightengan_rgb = cv2.cvtColor(enlightengan_img, cv2.COLOR_BGR2RGB)
            
            # 如果找不到EnlightenGAN增强图像，报错
            if enlightengan_img is None:
                print(f"❌ EnlightenGAN增强图像不存在: {gan_path}")
                fail_count += 1
                continue
            
            use_original_for_gan = False
            
            # 创建对比图（1行3列）
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # (a) 原始低光照图像 + Baseline 检测结果
            if 'baseline' in models:
                baseline_results = models['baseline'](original_img, verbose=False)
                original_with_det = draw_detections(original_rgb, baseline_results)
                axes[0].imshow(original_with_det)
            else:
                axes[0].imshow(original_rgb)
            
            axes[0].set_title('(a) 原始低光照图像\n+ Baseline 检测结果', 
                            fontsize=13, fontweight='bold', pad=10)
            axes[0].axis('off')
            
            # 计算检测数量
            if 'baseline' in models:
                det_count = len(baseline_results[0].boxes) if baseline_results and len(baseline_results) > 0 and baseline_results[0].boxes is not None else 0
                axes[0].text(0.5, -0.05, f'检测数量: {det_count}', 
                           transform=axes[0].transAxes, ha='center', fontsize=10)
            
            # (b) 传统增强后图像 + 检测结果
            if traditional_img is not None:
                if 'traditional' in models:
                    traditional_results = models['traditional'](traditional_img, verbose=False)
                    traditional_with_det = draw_detections(traditional_rgb, traditional_results)
                    axes[1].imshow(traditional_with_det)
                else:
                    axes[1].imshow(traditional_rgb)
                
                axes[1].set_title('(b) 传统增强后图像\n+ 检测结果', 
                                fontsize=13, fontweight='bold', color='green', pad=10)
                axes[1].axis('off')
                
                if 'traditional' in models:
                    det_count = len(traditional_results[0].boxes) if traditional_results and len(traditional_results) > 0 and traditional_results[0].boxes is not None else 0
                    axes[1].text(0.5, -0.05, f'检测数量: {det_count}', 
                               transform=axes[1].transAxes, ha='center', fontsize=10, color='green')
            else:
                axes[1].text(0.5, 0.5, '传统增强图像\n不存在', 
                           transform=axes[1].transAxes, ha='center', va='center',
                           fontsize=12, color='red')
                axes[1].axis('off')
            
            # (c) EnlightenGAN 增强后图像 + 检测结果
            if 'enlightengan' in models:
                enlightengan_results = models['enlightengan'](enlightengan_img, verbose=False)
                enlightengan_with_det = draw_detections(enlightengan_rgb, enlightengan_results)
                axes[2].imshow(enlightengan_with_det)
            elif 'baseline' in models:
                # 如果没有EnlightenGAN模型，使用baseline模型检测增强后的图像
                enlightengan_results = models['baseline'](enlightengan_img, verbose=False)
                enlightengan_with_det = draw_detections(enlightengan_rgb, enlightengan_results)
                axes[2].imshow(enlightengan_with_det)
            else:
                axes[2].imshow(enlightengan_rgb)
            
            axes[2].set_title('(c) EnlightenGAN 增强后图像\n+ 检测结果', 
                            fontsize=13, fontweight='bold', color='blue', pad=10)
            axes[2].axis('off')
            
            if 'enlightengan' in models or 'baseline' in models:
                det_count = len(enlightengan_results[0].boxes) if enlightengan_results and len(enlightengan_results) > 0 and enlightengan_results[0].boxes is not None else 0
                model_name = 'EnlightenGAN' if 'enlightengan' in models else 'Baseline'
                axes[2].text(0.5, -0.05, f'检测数量: {det_count} ({model_name})', 
                           transform=axes[2].transAxes, ha='center', fontsize=10, color='blue')
            
            plt.suptitle(f'Detection Comparison - {img_path.stem}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # 保存图像
            output_path = output_dir / f"detection_comparison_{i:02d}_{img_path.stem}.png"
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 已保存")
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
    print(f"\n可用于PPT的对比图：")
    for i in range(1, min(success_count + 1, 6)):
        print(f"  • detection_comparison_{i:02d}_*.png")
    print()

if __name__ == '__main__':
    main()

