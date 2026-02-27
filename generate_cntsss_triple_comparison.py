#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成CNTSSS夜间场景的三联检测对比图
包含：Baseline / 温和增强 / EnlightenGAN 三种结果
适合插入Word文档
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
    print("  生成CNTSSS夜间场景三联检测对比图")
    print("  (Baseline / 温和增强 / EnlightenGAN)")
    print("="*70)
    
    # 路径配置 - CNTSSS数据集
    original_dir = Path("data/CNTSSS/test/images")
    mild_enhanced_dir = Path("data/CNTSSS_mild_enhanced/test/images")
    enlightengan_dir = Path("data/CNTSSS_enlightengan_mild/test/images")
    
    # 模型路径
    baseline_model_path = Path("experiments/cntsss_baseline/run/weights/best.pt")
    mild_model_path = Path("experiments/cntsss_mild_enhanced/run/weights/best.pt")
    enlightengan_model_path = Path("experiments/cntsss_enlightengan_mild_enhanced/run/weights/best.pt")
    
    output_dir = Path("results/cntsss_triple_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查目录
    if not original_dir.exists():
        print(f"❌ 原始图像目录不存在: {original_dir}")
        return
    
    if not mild_enhanced_dir.exists():
        print(f"❌ 温和增强图像目录不存在: {mild_enhanced_dir}")
        return
    
    if not enlightengan_dir.exists():
        print(f"❌ EnlightenGAN增强图像目录不存在: {enlightengan_dir}")
        return
    
    # 检查模型文件
    models_to_load = {}
    if baseline_model_path.exists():
        models_to_load['baseline'] = baseline_model_path
    else:
        print(f"⚠️  Baseline模型不存在: {baseline_model_path}")
    
    if mild_model_path.exists():
        models_to_load['mild'] = mild_model_path
    else:
        print(f"⚠️  温和增强模型不存在: {mild_model_path}")
    
    if enlightengan_model_path.exists():
        models_to_load['enlightengan'] = enlightengan_model_path
    else:
        print(f"⚠️  EnlightenGAN模型不存在: {enlightengan_model_path}")
    
    if len(models_to_load) == 0:
        print("\n❌ 没有找到任何模型文件")
        print("   将使用Baseline模型检测所有图像")
        if baseline_model_path.exists():
            models_to_load['baseline'] = baseline_model_path
        else:
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
    all_images = sorted(original_dir.glob("*.jpg")) + sorted(original_dir.glob("*.png"))
    
    if len(all_images) == 0:
        print("❌ 没有找到图像")
        return
    
    # 随机选择几张图像，然后挑选检测效果最好的
    num_candidates = min(10, len(all_images))
    candidate_images = random.sample(all_images, num_candidates)
    
    print(f"\n🎲 从 {num_candidates} 张候选图像中选择最佳对比图...")
    
    best_image = None
    best_score = -1
    
    # 快速评估每张图像
    for img_path in candidate_images:
        try:
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                continue
            
            # 使用baseline模型快速检测
            if 'baseline' in models:
                results = models['baseline'](original_img, verbose=False)
                det_count = len(results[0].boxes) if results and len(results) > 0 and results[0].boxes is not None else 0
                # 选择检测数量适中的图像（2-5个目标）
                if 2 <= det_count <= 5:
                    score = det_count
                    if score > best_score:
                        best_score = score
                        best_image = img_path
        except:
            continue
    
    # 如果没有找到合适的，选择第一张
    if best_image is None:
        best_image = candidate_images[0]
        print(f"  使用随机选择的图像: {best_image.name}")
    else:
        print(f"  ✓ 选择最佳图像: {best_image.name} (检测数量: {best_score})")
    
    print(f"📂 输出目录: {output_dir}\n")
    
    try:
        # 读取原始图像
        original_img = cv2.imread(str(best_image))
        if original_img is None:
            print("❌ 原始图像读取失败")
            return
        
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 读取温和增强图像
        mild_path = mild_enhanced_dir / best_image.name
        if not mild_path.exists():
            print(f"❌ 温和增强图像不存在: {mild_path}")
            return
        
        mild_img = cv2.imread(str(mild_path))
        if mild_img is None:
            print("❌ 温和增强图像读取失败")
            return
        
        mild_rgb = cv2.cvtColor(mild_img, cv2.COLOR_BGR2RGB)
        
        # 读取EnlightenGAN增强图像
        gan_path = enlightengan_dir / best_image.name
        if not gan_path.exists():
            print(f"❌ EnlightenGAN增强图像不存在: {gan_path}")
            return
        
        enlightengan_img = cv2.imread(str(gan_path))
        if enlightengan_img is None:
            print("❌ EnlightenGAN增强图像读取失败")
            return
        
        enlightengan_rgb = cv2.cvtColor(enlightengan_img, cv2.COLOR_BGR2RGB)
        
        # 使用对应模型检测
        # (a) Baseline检测原始图像
        if 'baseline' in models:
            baseline_results = models['baseline'](original_img, verbose=False)
            original_with_det = draw_detections(original_rgb, baseline_results, color=(0, 255, 0))
        else:
            original_with_det = original_rgb
        
        baseline_det_count = len(baseline_results[0].boxes) if 'baseline' in models and baseline_results and len(baseline_results) > 0 and baseline_results[0].boxes is not None else 0
        
        # (b) 温和增强模型检测温和增强图像
        if 'mild' in models:
            mild_results = models['mild'](mild_img, verbose=False)
            mild_with_det = draw_detections(mild_rgb, mild_results, color=(0, 200, 255))
        elif 'baseline' in models:
            mild_results = models['baseline'](mild_img, verbose=False)
            mild_with_det = draw_detections(mild_rgb, mild_results, color=(0, 200, 255))
        else:
            mild_with_det = mild_rgb
        
        mild_det_count = len(mild_results[0].boxes) if ('mild' in models or 'baseline' in models) and mild_results and len(mild_results) > 0 and mild_results[0].boxes is not None else 0
        
        # (c) EnlightenGAN模型检测增强图像
        if 'enlightengan' in models:
            enlightengan_results = models['enlightengan'](enlightengan_img, verbose=False)
            enlightengan_with_det = draw_detections(enlightengan_rgb, enlightengan_results, color=(255, 0, 0))
        elif 'baseline' in models:
            enlightengan_results = models['baseline'](enlightengan_img, verbose=False)
            enlightengan_with_det = draw_detections(enlightengan_rgb, enlightengan_results, color=(255, 0, 0))
        else:
            enlightengan_with_det = enlightengan_rgb
        
        enlightengan_det_count = len(enlightengan_results[0].boxes) if ('enlightengan' in models or 'baseline' in models) and enlightengan_results and len(enlightengan_results) > 0 and enlightengan_results[0].boxes is not None else 0
        
        # 创建三联对比图（1行3列，适合Word）
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # (a) Baseline检测结果
        axes[0].imshow(original_with_det)
        axes[0].set_title('(a) Baseline\n(mAP@0.5: 67.89%)', 
                        fontsize=13, fontweight='bold', pad=10)
        axes[0].axis('off')
        axes[0].text(0.5, -0.05, f'检测数量: {baseline_det_count}', 
                   transform=axes[0].transAxes, ha='center', fontsize=11, color='green')
        
        # (b) 温和增强检测结果
        axes[1].imshow(mild_with_det)
        axes[1].set_title('(b) 温和增强\n(mAP@0.5: 67.09%)', 
                        fontsize=13, fontweight='bold', color='orange', pad=10)
        axes[1].axis('off')
        axes[1].text(0.5, -0.05, f'检测数量: {mild_det_count}', 
                   transform=axes[1].transAxes, ha='center', fontsize=11, color='orange')
        
        # (c) EnlightenGAN检测结果
        axes[2].imshow(enlightengan_with_det)
        axes[2].set_title('(c) EnlightenGAN\n(mAP@0.5: 41.30%)', 
                        fontsize=13, fontweight='bold', color='red', pad=10)
        axes[2].axis('off')
        axes[2].text(0.5, -0.05, f'检测数量: {enlightengan_det_count}', 
                   transform=axes[2].transAxes, ha='center', fontsize=11, color='red')
        
        # 添加总标题
        plt.suptitle('CNTSSS夜间场景检测结果对比', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 添加说明文字
        fig.text(0.5, 0.02, 
                '真实夜间场景下，Baseline和温和增强方法表现接近，EnlightenGAN性能显著下降',
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08, top=0.92)
        
        # 保存图像（高分辨率，适合Word）
        output_path = output_dir / f"cntsss_triple_comparison_{best_image.stem}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 三联对比图已保存: {output_path}")
        
        # 也保存为PDF格式（矢量图，更适合Word）
        output_path_pdf = output_dir / f"cntsss_triple_comparison_{best_image.stem}.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
        print(f"✅ 三联对比图已保存: {output_path_pdf}")
        
        plt.close()
        
        print(f"\n{'='*70}")
        print(f"✅ 三联对比图生成完成！")
        print(f"{'='*70}\n")
        print(f"图像: {best_image.name}")
        print(f"检测数量: Baseline={baseline_det_count}, 温和增强={mild_det_count}, EnlightenGAN={enlightengan_det_count}")
        print(f"\n📂 结果保存在: {output_dir}/")
        print(f"\n💡 适合插入Word文档，建议使用PDF格式（矢量图，清晰度更高）")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

