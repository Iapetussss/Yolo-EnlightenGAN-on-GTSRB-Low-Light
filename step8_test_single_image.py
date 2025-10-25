"""
步骤 8: 测试单张图像
在单张图像上测试模型，并可视化结果
"""

import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

print("=" * 60)
print("🖼️  步骤 8: 测试单张图像")
print("=" * 60)

# 查找训练好的模型
model_config = Path(__file__).parent / 'trained_model_path.txt'

if model_config.exists():
    with open(model_config, 'r') as f:
        model_path = Path(f.read().strip())
    print(f"\n✅ 使用训练好的模型: {model_path}")
else:
    print("\n请输入模型路径:")
    model_path = Path(input("模型路径: ").strip())

if not model_path.exists():
    print(f"\n❌ 模型文件不存在: {model_path}")
    sys.exit(1)

# 查找增强数据集
enhanced_config = Path(__file__).parent / 'enhanced_dataset_path.txt'

if enhanced_config.exists():
    with open(enhanced_config, 'r') as f:
        data_root = Path(f.read().strip())
    
    # 从测试集中随机选择一张图片
    test_images = list((data_root / 'test').glob('*.png'))
    
    if test_images:
        import random
        default_image = random.choice(test_images)
        print(f"\n随机选择测试图像: {default_image.name}")
        print("如果要使用其他图像，请输入路径")
        
        custom_image = input("图像路径 (直接按 Enter 使用随机图像): ").strip()
        
        if custom_image:
            test_image = Path(custom_image)
        else:
            test_image = default_image
    else:
        print("\n未找到测试图像，请手动输入:")
        test_image = Path(input("图像路径: ").strip())
else:
    print("\n请输入要测试的图像路径:")
    test_image = Path(input("图像路径: ").strip())

if not test_image.exists():
    print(f"\n❌ 图像文件不存在: {test_image}")
    sys.exit(1)

print(f"\n✅ 测试图像: {test_image}")

# 置信度阈值
conf_str = input("\n置信度阈值 (0.0-1.0，默认 0.25): ").strip()
try:
    conf = float(conf_str)
except:
    conf = 0.25

print(f"✅ 置信度阈值: {conf}")

# 配置文件
yaml_path = Path(__file__).parent.parent / 'traffic_signs.yaml'

# 开始测试
print("\n" + "=" * 60)
print("开始测试...")
print("=" * 60)

try:
    from enlightened_gtsrb import GTSRBEnlightenGANDetector
    
    # 创建检测器
    detector = GTSRBEnlightenGANDetector(config_path=str(yaml_path))
    
    # 加载模型
    print("\n加载模型...")
    detector.setup_yolov8(str(model_path))
    
    # 预测
    print("正在预测...")
    results = detector.predict(str(test_image), conf=conf, save=False)
    
    # 获取预测结果
    result = results[0]
    
    # 显示检测信息
    print("\n" + "=" * 60)
    print("检测结果:")
    print("=" * 60)
    
    if len(result.boxes) > 0:
        print(f"\n检测到 {len(result.boxes)} 个交通标志:\n")
        
        # 读取类别名称
        class_names = result.names
        
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = class_names[cls]
            
            print(f"  {i+1}. {class_name}")
            print(f"     置信度: {conf_score:.2%}")
            print(f"     位置: {box.xyxy[0].tolist()}")
            print()
    else:
        print("\n⚠️  未检测到交通标志")
        print("   可能原因:")
        print("   1. 置信度阈值太高")
        print("   2. 图像中确实没有交通标志")
        print("   3. 模型训练不足")
    
    # 可视化
    print("=" * 60)
    print("生成可视化结果...")
    print("=" * 60)
    
    # 读取原始图像
    original = cv2.imread(str(test_image))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # 获取预测结果图像
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('原始图像', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(annotated_rgb)
    axes[1].set_title(f'检测结果 (检测到 {len(result.boxes)} 个标志)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'result_{test_image.stem}.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化结果已保存到: {output_file}")
    
    # 显示图像
    print("\n正在显示结果...")
    plt.show()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    
    print("\n你可以:")
    print("  1. 测试更多图像: 重新运行此脚本")
    print("  2. 调整置信度阈值看看效果变化")
    print("  3. 生成完整的对比报告: python visualize_comparison.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 测试过程中出现错误:")
    print("=" * 60)
    print(str(e))
    import traceback
    traceback.print_exc()
    
    sys.exit(1)

