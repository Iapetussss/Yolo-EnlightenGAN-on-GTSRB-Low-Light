#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验4：温和增强 + YOLOv8 训练
使用温和参数（Gamma 1.15 + CLAHE 1.5）增强后的图像训练
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import yaml
from datetime import datetime

def main():
    print("="*70)
    print("  实验 4: 温和增强 + YOLOv8 训练")
    print("="*70)
    
    # 配置路径
    data_yaml = project_root / "configs" / "mild_dataset.yaml"
    model_path = project_root / "models" / "yolov8" / "yolov8n.pt"
    output_base = project_root / "experiments" / "exp4_mild"
    
    # 检查配置文件
    if not data_yaml.exists():
        print(f"\n❌ 配置文件不存在: {data_yaml}")
        print(f"   请先运行: python batch_enhance_adjustable.py")
        print(f"   并选择 [1] 温和增强")
        return
    
    # 检查模型
    if not model_path.exists():
        print(f"\n❌ 模型文件不存在: {model_path}")
        print(f"   YOLO 会自动下载...")
    
    print(f"\n📋 实验配置:")
    print(f"  模型: YOLOv8n")
    print(f"  数据: 温和增强后的图像")
    print(f"  方法: Gamma 1.15 + CLAHE 1.5")
    print(f"  配置: {data_yaml}")
    print(f"  输出: {output_base}")
    
    # 训练参数（与 baseline 一致）
    print(f"\n⚙️  训练参数:")
    epochs = 20
    batch = 16  # 与 baseline 保持一致
    imgsz = 640
    device = 0
    
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Device: GPU {device}")
    
    # 确认
    print(f"\n⏱️  预计训练时间: 10-13 小时")
    response = input(f"\n开始训练？[y/n]: ").strip().lower()
    
    if response != 'y':
        print("❌ 训练已取消")
        return
    
    # 创建输出目录
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"  训练开始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    try:
        # 加载模型
        model = YOLO(str(model_path))
        
        # 训练
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=str(output_base),
            name='run',
            exist_ok=True,
            patience=50,
            save=True,
            plots=True,
            workers=2,
            amp=False
        )
        
        # 计算训练时间
        end_time = datetime.now()
        duration = end_time - start_time
        
        # 获取最终指标
        results_csv = output_base / "run" / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            last_row = df.iloc[-1]
            
            final_map50 = last_row['metrics/mAP50(B)']
            final_map = last_row['metrics/mAP50-95(B)']
            final_precision = last_row['metrics/precision(B)']
            final_recall = last_row['metrics/recall(B)']
        else:
            final_map50 = 0
            final_map = 0
            final_precision = 0
            final_recall = 0
        
        # 保存实验信息
        experiment_info = {
            'experiment_name': '温和增强 + YOLOv8',
            'model': 'YOLOv8n',
            'enhancement_method': 'Mild Enhancement (Gamma 1.15 + CLAHE 1.5)',
            'data_source': 'Mild enhanced images',
            'training_params': {
                'epochs': epochs,
                'batch_size': batch,
                'image_size': imgsz,
                'device': f'cuda:{device}',
                'workers': 2,
                'amp': False
            },
            'training_time': str(duration),
            'final_metrics': {
                'mAP@0.5': float(final_map50),
                'mAP@0.5:0.95': float(final_map),
                'Precision': float(final_precision),
                'Recall': float(final_recall)
            },
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        info_path = output_base / "experiment_info.yaml"
        with open(info_path, 'w', encoding='utf-8') as f:
            yaml.dump(experiment_info, f, allow_unicode=True)
        
        # 打印结果
        print(f"\n{'='*70}")
        print(f"                          ✅ 实验 4 训练完成！")
        print(f"{'='*70}\n")
        print(f"训练时间: {duration}\n")
        print(f"结果保存在:")
        print(f"  {output_base / 'run'}/\n")
        print(f"关键指标:")
        print(f"  mAP@0.5:      {final_map50:.4f}")
        print(f"  mAP@0.5:0.95: {final_map:.4f}")
        print(f"  Precision:    {final_precision:.4f}")
        print(f"  Recall:       {final_recall:.4f}\n")
        print(f"实验信息已保存: {info_path}\n")
        
        print(f"{'='*70}")
        print(f"四组实验对比")
        print(f"{'='*70}\n")
        print(f"1. Baseline (纯低光照):     mAP = 70.4%")
        print(f"2. 传统方法增强:            mAP = 70.1%")
        print(f"3. 温和增强:                mAP = {final_map50*100:.1f}%")
        print(f"4. EnlightenGAN增强:        mAP = 39.7%")
        print(f"\n{'='*70}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  训练被用户中断")
        print(f"部分结果可能已保存在: {output_base}")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

