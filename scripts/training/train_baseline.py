"""
实验 1: Baseline - 纯 YOLOv8（无图像增强）

目标：
- 建立性能基线
- 验证低光照对检测性能的影响
- 预期 mAP: 60-70%
"""

import sys
from pathlib import Path
import yaml
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parents[2]))

def main():
    """主函数"""
    
    print("\n" + "=" * 70)
    print("  实验 1: Baseline (纯 YOLOv8，无增强)".center(70))
    print("=" * 70)
    
    print("\n实验说明：")
    print("  • 使用低光照图像直接训练 YOLOv8")
    print("  • 不进行任何图像增强")
    print("  • 目标：建立性能基线")
    print("  • 预期 mAP@0.5: 60-70%")
    
    # 检查数据
    print("\n" + "=" * 70)
    print("检查数据...")
    print("=" * 70)
    
    # 检查 Baseline 专用数据
    baseline_data = Path('data/baseline_lowlight_dataset')
    
    if not baseline_data.exists():
        print("❌ 未找到 Baseline 数据集")
        print("\n⚠️  Baseline 实验需要【纯低光照图像】（无任何增强）")
        print("\n请先运行:")
        print("   python scripts/preprocessing/create_pure_lowlight.py")
        print("\n该脚本将:")
        print("  • 使用 Gamma 变换生成纯低光照图像")
        print("  • 不进行任何增强处理")
        print("  • 创建 YOLO 格式数据集")
        print("  • 预计耗时: 30-60 分钟")
        
        response = input("\n是否现在运行？(y/N): ").strip().lower()
        if response == 'y':
            print("\n启动数据生成...")
            import subprocess
            result = subprocess.run(
                [sys.executable, 'scripts/preprocessing/create_pure_lowlight.py'],
                cwd=Path(__file__).parents[2]
            )
            if result.returncode != 0:
                print("\n❌ 数据生成失败")
                sys.exit(1)
            # 重新检查
            if not baseline_data.exists():
                print("\n❌ 数据集仍未找到")
                sys.exit(1)
        else:
            print("已取消")
            sys.exit(0)
    
    data_root = baseline_data
    print(f"✅ 找到Baseline数据集: {data_root}")
    
    # 统计数据
    for split in ['train', 'val', 'test']:
        img_dir = data_root / 'images' / split
        if img_dir.exists():
            img_count = len(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
            print(f"  {split.upper():<6}: {img_count:>6} 张图像")
    
    # 创建配置文件
    print("\n" + "=" * 70)
    print("创建实验配置...")
    print("=" * 70)
    
    config = {
        'path': str(data_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 43,
        'names': [
            "speed_20", "speed_30", "speed_50", "speed_60", "speed_70", "speed_80",
            "speed_80_end", "speed_100", "speed_120", "no_overtaking", "no_overtaking_trucks",
            "priority_at_next_intersection", "priority_road", "give_way", "stop", "no_entry",
            "no_entry_trucks", "no_entry_one_way", "general_caution", "dangerous_curve_left",
            "dangerous_curve_right", "double_curve", "bumpy_road", "slippery_road", 
            "road_narrows_right", "road_works", "traffic_signals", "pedestrians", 
            "children_crossing", "bicycles_crossing", "ice_or_snow", "wild_animals_crossing",
            "end_of_all_speed_and_overtaking_limits", "turn_right_ahead", "turn_left_ahead",
            "ahead_only", "go_straight_or_right", "go_straight_or_left", "keep_right",
            "keep_left", "roundabout_mandatory", "end_of_no_overtaking", 
            "end_of_no_overtaking_trucks"
        ]
    }
    
    # 保存配置
    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / 'exp1_baseline.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 配置已保存: {config_path}")
    
    # 训练参数
    print("\n" + "=" * 70)
    print("训练参数设置...")
    print("=" * 70)
    
    print("\n请输入训练参数（直接回车使用默认值）：")
    
    # Epochs
    epochs_input = input(f"训练轮数 Epochs (默认 20): ").strip()
    epochs = int(epochs_input) if epochs_input else 20
    
    # Batch size
    print("\nBatch Size 建议:")
    print("  • RTX 4060 (8GB): 使用 2 ⭐ (最稳定)")
    print("  • 如果想更快可尝试 4，但可能OOM")
    batch_input = input(f"Batch Size (默认 2): ").strip()
    batch = int(batch_input) if batch_input else 2
    
    # Device
    device_input = input(f"设备 (0=GPU, cpu=CPU, 默认 0): ").strip()
    device = device_input if device_input else '0'
    
    print(f"\n训练配置:")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch Size: {batch}")
    print(f"  Device:     {device}")
    print(f"  Image Size: 640")
    print(f"  Model:      YOLOv8n")
    
    response = input("\n开始训练？(y/N): ").strip().lower()
    if response != 'y':
        print("已取消")
        sys.exit(0)
    
    # 开始训练
    print("\n" + "=" * 70)
    print("  开始训练 实验 1: Baseline".center(70))
    print("=" * 70)
    
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model_path = Path('models/yolov8/yolov8n.pt')
        if not model_path.exists():
            model_path = Path('yolov8n.pt')
        
        if not model_path.exists():
            print("\n下载 YOLOv8n 模型...")
            model = YOLO('yolov8n.pt')
        else:
            model = YOLO(str(model_path))
        
        print(f"✅ 模型加载成功: YOLOv8n")
        
        # 训练
        print(f"\n{'=' * 70}")
        print("开始训练...")
        print(f"{'=' * 70}\n")
        
        start_time = datetime.now()
        
        results = model.train(
            data=str(config_path),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device=device,
            workers=0,  # 降低到0以节省显存
            cache=False,  # 不缓存数据到内存
            project='experiments/exp1_baseline',
            name='run',
            plots=True,
            save=True,
            verbose=True,
            amp=False,  # 禁用 AMP 以节省显存
            
            # 不使用任何数据增强（纯baseline）
            augment=False,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # 训练完成
        print(f"\n{'=' * 70}")
        print("  ✅ 实验 1 训练完成！".center(70))
        print(f"{'=' * 70}")
        
        print(f"\n训练时间: {training_time}")
        print(f"\n结果保存在:")
        print(f"  experiments/exp1_baseline/run/")
        
        # 显示关键指标
        print(f"\n关键指标:")
        results_dict = results.results_dict
        print(f"  mAP@0.5:      {results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:0.95: {results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision:    {results_dict.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall:       {results_dict.get('metrics/recall(B)', 0):.4f}")
        
        # 保存实验信息
        exp_info = {
            'experiment': 'exp1_baseline',
            'description': '纯 YOLOv8，无图像增强',
            'data': str(data_root),
            'model': 'YOLOv8n',
            'enhancement': 'None',
            'epochs': epochs,
            'batch_size': batch,
            'training_time': str(training_time),
            'results': {
                'mAP50': float(results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results_dict.get('metrics/recall(B)', 0)),
            }
        }
        
        exp_info_path = Path('experiments/exp1_baseline/experiment_info.yaml')
        with open(exp_info_path, 'w') as f:
            yaml.dump(exp_info, f, default_flow_style=False)
        
        print(f"\n实验信息已保存: {exp_info_path}")
        
        # 下一步
        print(f"\n{'=' * 70}")
        print("下一步：")
        print(f"{'=' * 70}")
        
        print("\n1. 查看训练结果:")
        print("   experiments/exp1_baseline/run/results.png")
        print("   experiments/exp1_baseline/run/confusion_matrix.png")
        
        print("\n2. 运行实验 2 (Traditional Enhancement):")
        print("   python scripts/training/train_traditional.py")
        
        print("\n3. 评估模型:")
        print("   python scripts/evaluation/evaluate_model.py \\")
        print("       --model experiments/exp1_baseline/run/weights/best.pt \\")
        print("       --data configs/exp1_baseline.yaml")
        
        print(f"\n{'=' * 70}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print("   已保存的检查点在: experiments/exp1_baseline/run/")
        
    except Exception as e:
        print(f"\n{'=' * 70}")
        print("❌ 训练出错:")
        print(f"{'=' * 70}")
        print(str(e))
        import traceback
        traceback.print_exc()
        
        print("\n可能的原因:")
        print("  1. 显存/内存不足 → 减小 batch size")
        print("  2. 数据路径错误 → 检查数据集位置")
        print("  3. CUDA 错误 → 尝试使用 device='cpu'")
        
        sys.exit(1)

if __name__ == '__main__':
    main()

