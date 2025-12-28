"""
快速测试单张图像
自动使用训练好的模型
"""

from pathlib import Path
import random
import sys

def find_best_model():
    """找到最好的训练模型"""
    models = list(Path('runs/train').glob('*/weights/best.pt'))
    
    if not models:
        print("❌ 未找到训练好的模型")
        print("请先完成训练: python step6_train_model.py")
        return None
    
    # 使用最新的模型
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    return latest_model

def find_test_image():
    """找一张测试图像"""
    possible_dirs = [
        Path('traffic_sign_data/enhanced_images/test'),
        Path('yolo_dataset/images/test'),
        Path('yolo_dataset/images/val'),
    ]
    
    for img_dir in possible_dirs:
        if img_dir.exists():
            images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
            if images:
                return random.choice(images)
    
    return None

def main():
    print("\n" + "=" * 70)
    print("  快速图像测试".center(70))
    print("=" * 70)
    
    # 1. 找模型
    print("\n🔍 寻找训练好的模型...")
    model_path = find_best_model()
    
    if not model_path:
        sys.exit(1)
    
    print(f"✅ 找到模型: {model_path}")
    print(f"   训练实验: {model_path.parent.parent.name}")
    
    # 2. 找测试图像
    print("\n🔍 选择测试图像...")
    test_image = find_test_image()
    
    if not test_image:
        print("❌ 未找到测试图像")
        sys.exit(1)
    
    print(f"✅ 测试图像: {test_image}")
    
    # 3. 加载模型和预测
    print("\n🚀 开始测试...")
    print("-" * 70)
    
    from ultralytics import YOLO
    import cv2
    import matplotlib.pyplot as plt
    
    # 加载模型
    model = YOLO(str(model_path))
    
    # 预测
    results = model.predict(
        source=str(test_image),
        conf=0.25,
        save=False,
        verbose=False
    )
    
    # 获取结果
    result = results[0]
    
    print(f"\n📊 检测结果:")
    print(f"   检测到 {len(result.boxes)} 个目标")
    
    if len(result.boxes) > 0:
        print("\n详细信息:")
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # 获取类别名称
            class_name = result.names[cls]
            print(f"   {i+1}. {class_name}: {conf:.2%} 置信度")
    else:
        print("   ⚠️  未检测到任何目标")
    
    # 4. 可视化
    print("\n🎨 生成可视化...")
    
    # 读取原图
    img = cv2.imread(str(test_image))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 获取标注后的图像
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(annotated_rgb)
    axes[1].set_title(f'Detection Result ({len(result.boxes)} objects)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_path = 'quick_test_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 结果已保存: {output_path}")
    
    # 显示
    try:
        plt.show()
    except:
        pass
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)
    
    print("\n💡 想测试更多图像？再次运行:")
    print("   python quick_test_image.py")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

