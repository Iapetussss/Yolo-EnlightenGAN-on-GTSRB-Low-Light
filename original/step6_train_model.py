"""
步骤 6: 训练 YOLOv8 模型
使用增强后的数据训练交通标志检测模型
"""

import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 步骤 6: 训练 YOLOv8 模型")
    print("=" * 60)

    # 检查配置文件
    yaml_path = Path(__file__).parent / 'traffic_signs_dataset.yaml'

    if not yaml_path.exists():
        print(f"\n❌ 错误: 配置文件不存在: {yaml_path}")
        print("   请确保已完成前面的步骤")
        sys.exit(1)

    print(f"\n✅ 配置文件: {yaml_path}")

    # 显示配置内容
    print("\n当前配置:")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[:10]:  # 只显示前10行
            print(f"   {line.rstrip()}")

    # 训练参数设置
    print("\n" + "=" * 60)
    print("训练参数设置:")
    print("=" * 60)

    print("\n1. 选择模型大小:")
    print("   - yolov8n.pt (Nano): 最快，最小，精度稍低")
    print("   - yolov8s.pt (Small): 平衡速度和精度 (推荐)")
    print("   - yolov8m.pt (Medium): 较慢，精度较高")
    print("   - yolov8l.pt (Large): 很慢，精度很高")

    model_choice = input("\n请选择模型 (n/s/m/l，默认 n): ").strip().lower()
    if model_choice not in ['n', 's', 'm', 'l']:
        model_choice = 'n'

    model_path = f'yolov8{model_choice}.pt'
    print(f"✅ 选择的模型: {model_path}")

    # 训练轮数
    print("\n2. 训练轮数 (epochs):")
    print("   - 10-20: 快速测试")
    print("   - 50: 初步训练 (推荐新手先试试)")
    print("   - 100: 标准训练")
    print("   - 200+: 充分训练")

    epochs = input("\n请输入训练轮数 (默认 50): ").strip()
    try:
        epochs = int(epochs)
    except:
        epochs = 50

    print(f"✅ 训练轮数: {epochs}")

    # 批次大小
    print("\n3. 批次大小 (batch size):")
    print("   - 1: 最小显存占用 (2-4GB)")
    print("   - 2: 适合小显存 (4-6GB) [推荐]")
    print("   - 4: 标准配置 (8GB 显存)")
    print("   - 8+: 大显存 (12GB+ 显存)")

    batch = input("\n请输入批次大小 (默认 2): ").strip()
    try:
        batch = int(batch)
    except:
        batch = 2

    print(f"✅ 批次大小: {batch}")

    # 设备选择
    print("\n4. 设备选择:")
    print("   - 0: 使用 GPU 0 (如果有)")
    print("   - cpu: 使用 CPU (慢但稳定)")

    device = input("\n请输入设备 (0/cpu，默认 0): ").strip()
    if not device:
        device = '0'

    print(f"✅ 设备: {device}")

    # 确认
    print("\n" + "=" * 60)
    print("训练配置总结:")
    print("=" * 60)
    print(f"  模型: {model_path}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch}")
    print(f"  设备: {device}")
    print(f"  配置文件: {yaml_path}")
    print("\n⚠️  注意:")
    print(f"  - 预计训练时间: {epochs * 2} - {epochs * 10} 分钟")
    print("  - 训练过程中可以按 Ctrl+C 中断")
    print("  - 结果会保存在 runs/train/gtsrb_enlightengan/")

    response = input("\n是否开始训练? (输入 yes 继续): ").strip().lower()

    if response != 'yes':
        print("\n❌ 用户取消训练")
        sys.exit(0)

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    try:
        from enlightened_gtsrb import GTSRBEnlightenGANDetector
        
        # 创建检测器
        detector = GTSRBEnlightenGANDetector(config_path=str(yaml_path))
        
        # 加载模型
        print(f"\n加载模型: {model_path}")
        detector.setup_yolov8(model_path)
        
        # 开始训练
        print(f"\n开始训练 {epochs} 轮...")
        print("=" * 60)
        
        results = detector.train_yolov8(
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device=device,
            workers=2  # 减少worker数量以节省显存
        )
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        
        # 找到最佳模型
        best_model_path = Path('runs/train/gtsrb_enlightengan/weights/best.pt')
        last_model_path = Path('runs/train/gtsrb_enlightengan/weights/last.pt')
        
        if best_model_path.exists():
            print(f"\n✅ 最佳模型已保存: {best_model_path}")
            
            # 保存模型路径
            model_config = Path(__file__).parent / 'trained_model_path.txt'
            with open(model_config, 'w') as f:
                f.write(str(best_model_path.absolute()))
            print(f"✅ 模型路径已保存")
        
        if last_model_path.exists():
            print(f"✅ 最后模型已保存: {last_model_path}")
        
        # 显示结果
        results_dir = Path('runs/train/gtsrb_enlightengan')
        if results_dir.exists():
            print(f"\n📊 训练结果目录: {results_dir}")
            print("\n在这个目录中你可以找到:")
            print("  - weights/best.pt: 最佳模型")
            print("  - weights/last.pt: 最后一轮模型")
            print("  - results.png: 训练曲线图")
            print("  - confusion_matrix.png: 混淆矩阵")
            print("  - 其他可视化结果")
        
        print("\n" + "=" * 60)
        print("下一步:")
        print("  1. 查看训练结果: 打开 runs/train/gtsrb_enlightengan/results.png")
        print("  2. 评估模型: python step7_evaluate_model.py")
        print("  3. 测试单张图像: python step8_test_single_image.py")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print("   已保存的模型和结果保留在 runs/train/ 目录中")
        print("   你可以稍后继续训练或使用当前的模型")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 训练过程中出现错误:")
        print("=" * 60)
        print(str(e))
        import traceback
        traceback.print_exc()
        
        print("\n可能的原因:")
        print("1. 显存/内存不足 → 减小 batch size")
        print("2. 数据路径错误 → 检查 traffic_signs.yaml")
        print("3. CUDA 错误 → 尝试使用 device='cpu'")
        print("4. 数据集问题 → 重新运行数据准备步骤")
        
        sys.exit(1)

if __name__ == '__main__':
    main()

