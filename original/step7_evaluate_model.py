"""
步骤 7: 评估训练好的模型
在验证集和测试集上评估模型性能
"""

import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("📊 步骤 7: 评估模型性能")
    print("=" * 60)

    # 自动查找最新的训练模型
    train_dir = Path('runs/train')
    if train_dir.exists():
        # 找到最新的训练目录
        train_runs = [d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith('gtsrb')]
        if train_runs:
            latest_run = max(train_runs, key=lambda x: x.stat().st_mtime)
            auto_model = latest_run / 'weights' / 'best.pt'
            if auto_model.exists():
                print(f"\n✅ 自动检测到最新模型: {auto_model}")
                use_auto = input("使用这个模型? (yes/no，默认 yes): ").strip().lower()
                if use_auto != 'no':
                    model_path = auto_model
                else:
                    model_path = input("请输入模型路径: ").strip()
            else:
                print("\n请输入训练好的模型路径:")
                print("例如: runs/train/gtsrb_enlightengan8/weights/best.pt")
                model_path = input("\n模型路径: ").strip()
        else:
            print("\n请输入训练好的模型路径:")
            print("例如: runs/train/gtsrb_enlightengan8/weights/best.pt")
            model_path = input("\n模型路径: ").strip()
    else:
        print("\n请输入训练好的模型路径:")
        print("例如: runs/train/gtsrb_enlightengan8/weights/best.pt")
        model_path = input("\n模型路径: ").strip()

    if not model_path:
        print("\n❌ 必须提供模型路径")
        sys.exit(1)

    model_path = Path(model_path)

    if not model_path.exists():
        print(f"\n❌ 模型文件不存在: {model_path}")
        sys.exit(1)

    # 检查配置文件
    yaml_path = Path(__file__).parent / 'traffic_signs_dataset.yaml'

    if not yaml_path.exists():
        # 尝试旧的配置文件
        yaml_path = Path(__file__).parent / 'traffic_signs.yaml'
        if not yaml_path.exists():
            print(f"\n❌ 配置文件不存在: {yaml_path}")
            sys.exit(1)

    print(f"\n✅ 模型路径: {model_path}")
    print(f"✅ 配置文件: {yaml_path}")

    # 选择评估数据集
    print("\n" + "=" * 60)
    print("选择评估数据集:")
    print("=" * 60)
    print("1. val - 验证集")
    print("2. test - 测试集")
    print("3. both - 两者都评估 (推荐)")

    choice = input("\n请选择 (1/2/3，默认 3): ").strip()

    if choice == '1':
        splits = ['val']
    elif choice == '2':
        splits = ['test']
    else:
        splits = ['val', 'test']

    # 设备选择
    device = input("\n使用设备 (0/cpu，默认 0): ").strip()
    if not device:
        device = '0'

    # 开始评估
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)

    try:
        from enlightened_gtsrb import GTSRBEnlightenGANDetector
        
        # 创建检测器
        detector = GTSRBEnlightenGANDetector(config_path=str(yaml_path))
        
        # 加载模型
        print(f"\n加载模型...")
        detector.setup_yolov8(str(model_path))
        
        results_dict = {}
        
        for split in splits:
            print(f"\n{'=' * 60}")
            print(f"在 {split.upper()} 集上评估...")
            print('=' * 60)
            
            results = detector.validate(split=split, device=device, workers=2)
            results_dict[split] = results
            
            print(f"\n✅ {split.upper()} 集评估完成")
        
        # 显示结果总结
        print("\n" + "=" * 60)
        print("📊 评估结果总结:")
        print("=" * 60)
        
        for split, results in results_dict.items():
            print(f"\n{split.upper()} 集:")
            if hasattr(results, 'box'):
                print(f"  mAP@0.5:      {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
                print(f"  mAP@0.5:0.95: {results.box.map:.4f} ({results.box.map*100:.2f}%)")
                print(f"  Precision:    {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
                print(f"  Recall:       {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
        
        print("\n" + "=" * 60)
        print("✅ 评估完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 评估过程中出现错误:")
        print("=" * 60)
        print(str(e))
        import traceback
        traceback.print_exc()
        
        print("\n可能的原因:")
        print("1. 模型文件损坏")
        print("2. 数据路径错误")
        print("3. 配置文件错误")
        
        sys.exit(1)

if __name__ == '__main__':
    main()
