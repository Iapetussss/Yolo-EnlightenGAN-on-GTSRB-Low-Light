#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNTSSS Enhanced: 增强后训练
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
    print("  CNTSSS Enhanced: 增强后训练")
    print("="*70)
    
    # 检查可用的配置文件
    configs = list((project_root / "configs").glob("cntsss_*_dataset.yaml"))
    # 排除 cntsss_dataset.yaml（Baseline用的）
    configs = [c for c in configs if c.name != "cntsss_dataset.yaml"]
    
    if len(configs) == 0:
        print("\n❌ 未找到增强数据集配置")
        print("请先运行批量增强脚本:")
        print("  python batch_enhance_cntsss.py")
        print("  或 python batch_enhance_cntsss_enlightengan.py")
        return
    elif len(configs) == 1:
        data_yaml = configs[0]
        method = data_yaml.stem.replace("cntsss_", "").replace("_dataset", "")
        print(f"\n找到配置: {data_yaml.name}")
    else:
        print("\n发现多个增强数据集，请选择：")
        for i, cfg in enumerate(configs, 1):
            print(f"  [{i}] {cfg.stem.replace('cntsss_', '').replace('_dataset', '')}")
        choice = input(f"\n选择 [1-{len(configs)}]: ").strip()
        try:
            idx = int(choice) - 1
            data_yaml = configs[idx]
            method = data_yaml.stem.replace("cntsss_", "").replace("_dataset", "")
        except:
            print("❌ 无效选择")
            return
    
    model_path = project_root / "models" / "yolov8" / "yolov8n.pt"
    output_base = project_root / "experiments" / f"cntsss_{method}_enhanced"
    
    print(f"\n📋 实验配置:")
    print(f"  数据集: CNTSSS ({method} 增强)")
    print(f"  配置: {data_yaml}")
    
    # 训练参数
    print(f"\n⚙️  训练参数:")
    epochs = 50
    batch = 16
    imgsz = 640
    device = 0
    
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    
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
            'experiment_name': f'CNTSSS {method} Enhanced',
            'dataset': 'CNTSSS (Real night-time scenes)',
            'model': 'YOLOv8n',
            'enhancement_method': f'{method} enhancement',
            'training_params': {
                'epochs': epochs,
                'batch_size': batch,
                'image_size': imgsz,
                'device': f'cuda:{device}',
                'workers': 4,
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
        print(f"  ✅ CNTSSS {method} Enhanced 训练完成！")
        print(f"{'='*70}\n")
        print(f"训练时间: {duration}\n")
        print(f"结果保存在: {output_base / 'run'}/\n")
        print(f"关键指标:")
        print(f"  mAP@0.5:      {final_map50:.4f} ({final_map50*100:.1f}%)")
        print(f"  mAP@0.5:0.95: {final_map:.4f} ({final_map*100:.1f}%)")
        print(f"  Precision:    {final_precision:.4f} ({final_precision*100:.1f}%)")
        print(f"  Recall:       {final_recall:.4f} ({final_recall*100:.1f}%)")
        print(f"\n{'='*70}")
        print("  实验对比")
        print(f"{'='*70}\n")
        print("查看 experiments/ 目录对比 Baseline vs Enhanced 结果")
        print()
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

