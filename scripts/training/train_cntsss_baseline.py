#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNTSSS Baseline: 真实夜间场景直接训练
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import yaml
from datetime import datetime

def main():
    print("="*70)
    print("  CNTSSS Baseline: 真实夜间场景训练")
    print("="*70)
    
    # 配置
    data_yaml = project_root / "configs" / "cntsss_dataset.yaml"
    model_path = project_root / "models" / "yolov8" / "yolov8n.pt"
    output_base = project_root / "experiments" / "cntsss_baseline"
    
    if not data_yaml.exists():
        print(f"\n❌ 配置文件不存在: {data_yaml}")
        return
    
    print(f"\n📋 实验配置:")
    print(f"  数据集: CNTSSS (真实夜间场景)")
    print(f"  训练集: 3276 张")
    print(f"  测试集: 786 张")
    print(f"  类别: 3 类")
    print(f"  配置: {data_yaml}")
    
    # 训练参数
    print(f"\n⚙️  训练参数:")
    epochs = 50  # 数据量小，多训练几轮
    batch = 16   # 数据量小，可以用大 batch
    imgsz = 640
    device = 0
    
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Device: GPU {device}")
    
    print(f"\n⏱️  预计训练时间: 2-3 小时")
    response = input(f"\n开始训练？[y/n]: ").strip().lower()
    
    if response != 'y':
        print("❌ 训练已取消")
        return
    
    output_base.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"  训练开始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    try:
        model = YOLO(str(model_path))
        
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
            workers=4,
            amp=False
        )
        
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
            final_map50 = final_map = final_precision = final_recall = 0
        
        # 保存实验信息
        experiment_info = {
            'experiment_name': 'CNTSSS Baseline',
            'dataset': 'CNTSSS (Real night-time scenes)',
            'model': 'YOLOv8n',
            'data_type': 'Original night-time images (no enhancement)',
            'training_params': {
                'epochs': epochs,
                'batch_size': batch,
                'image_size': imgsz,
                'device': f'cuda:{device}',
                'workers': 4,
                'amp': False
            },
            'dataset_stats': {
                'train_images': 3276,
                'test_images': 786,
                'num_classes': 3
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
        print(f"  ✅ CNTSSS Baseline 训练完成！")
        print(f"{'='*70}\n")
        print(f"训练时间: {duration}\n")
        print(f"结果保存在: {output_base / 'run'}/\n")
        print(f"关键指标:")
        print(f"  mAP@0.5:      {final_map50:.4f} ({final_map50*100:.1f}%)")
        print(f"  mAP@0.5:0.95: {final_map:.4f} ({final_map*100:.1f}%)")
        print(f"  Precision:    {final_precision:.4f} ({final_precision*100:.1f}%)")
        print(f"  Recall:       {final_recall:.4f} ({final_recall*100:.1f}%)\n")
        print(f"实验信息已保存: {info_path}\n")
        print(f"{'='*70}")
        print(f"下一步：运行增强实验进行对比")
        print(f"{'='*70}\n")
        print("1. 批量增强数据:")
        print("   python batch_enhance_cntsss.py\n")
        print("2. 训练增强模型:")
        print("   python scripts/training/train_cntsss_enhanced.py")
        print()
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

