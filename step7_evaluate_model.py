"""
步骤 7: 评估训练好的模型
在验证集和测试集上评估模型性能
"""

import sys
from pathlib import Path

print("=" * 60)
print("📊 步骤 7: 评估模型性能")
print("=" * 60)

# 查找训练好的模型
model_config = Path(__file__).parent / 'trained_model_path.txt'

if model_config.exists():
    with open(model_config, 'r') as f:
        default_model = f.read().strip()
    print(f"\n检测到训练好的模型: {default_model}")
    
    use_default = input("是否使用这个模型? (yes/no，默认 yes): ").strip().lower()
    
    if use_default != 'no':
        model_path = default_model
    else:
        model_path = input("请输入模型路径: ").strip()
else:
    print("\n未找到保存的模型路径")
    print("请输入训练好的模型路径:")
    print("例如: runs/train/gtsrb_enlightengan/weights/best.pt")
    model_path = input("\n模型路径: ").strip()

if not model_path:
    print("\n❌ 必须提供模型路径")
    sys.exit(1)

model_path = Path(model_path)

if not model_path.exists():
    print(f"\n❌ 模型文件不存在: {model_path}")
    sys.exit(1)

# 检查配置文件
yaml_path = Path(__file__).parent.parent / 'traffic_signs.yaml'

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
        
        results = detector.validate(split=split, device=device)
        results_dict[split] = results
        
        print(f"\n✅ {split.upper()} 集评估完成")
    
    # 显示结果总结
    print("\n" + "=" * 60)
    print("📊 评估结果总结:")
    print("=" * 60)
    
    for split, results in results_dict.items():
        print(f"\n{split.upper()} 集:")
        
        # YOLOv8 的结果对象包含多个指标
        if hasattr(results, 'box'):
            box = results.box
            print(f"  Precision: {box.mp:.4f}")  # mean precision
            print(f"  Recall:    {box.mr:.4f}")  # mean recall
            print(f"  mAP50:     {box.map50:.4f}")  # mAP at IoU=0.5
            print(f"  mAP50-95:  {box.map:.4f}")  # mAP at IoU=0.5:0.95
        else:
            print("  结果对象格式不同，请查看详细输出")
    
    # 结果保存位置
    results_dir = Path('runs/val')
    if results_dir.exists():
        latest_dir = max(results_dir.iterdir(), key=lambda x: x.stat().st_mtime)
        print(f"\n📁 详细结果已保存到: {latest_dir}")
        print("\n在这个目录中你可以找到:")
        print("  - confusion_matrix.png: 混淆矩阵")
        print("  - val_batch*_labels.jpg: 标签可视化")
        print("  - val_batch*_pred.jpg: 预测结果可视化")
    
    print("\n" + "=" * 60)
    print("✅ 评估完成！")
    print("=" * 60)
    
    print("\n下一步:")
    print("  - 测试单张图像: python step8_test_single_image.py")
    print("  - 可视化结果对比: python visualize_comparison.py")
    
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

