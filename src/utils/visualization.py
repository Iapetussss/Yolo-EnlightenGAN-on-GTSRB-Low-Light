"""
可视化对比工具
对比原始图像、低光照图像、增强图像和检测结果
"""

import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("🎨 可视化对比工具")
print("=" * 60)

# 查找模型
model_config = Path(__file__).parent / 'trained_model_path.txt'

if model_config.exists():
    with open(model_config, 'r') as f:
        model_path = Path(f.read().strip())
    print(f"\n✅ 使用模型: {model_path.name}")
else:
    print("\n请先训练模型: python step6_train_model.py")
    sys.exit(1)

if not model_path.exists():
    print(f"\n❌ 模型不存在: {model_path}")
    sys.exit(1)

# 查找数据集路径
original_config = Path(__file__).parent / 'converted_dataset_path.txt'
lowlight_config = Path(__file__).parent / 'lowlight_dataset_path.txt'
enhanced_config = Path(__file__).parent / 'enhanced_dataset_path.txt'

if not all([original_config.exists(), lowlight_config.exists(), enhanced_config.exists()]):
    print("\n❌ 数据集路径配置不完整")
    print("   请确保已完成所有数据准备步骤")
    sys.exit(1)

# 读取路径
with open(original_config, 'r') as f:
    original_root = Path(f.read().strip())
with open(lowlight_config, 'r') as f:
    lowlight_root = Path(f.read().strip())
with open(enhanced_config, 'r') as f:
    enhanced_root = Path(f.read().strip())

print(f"\n✅ 原始数据: {original_root.name}")
print(f"✅ 低光照数据: {lowlight_root.name}")
print(f"✅ 增强数据: {enhanced_root.name}")

# 选择几张图像进行对比
test_images_original = list((original_root / 'images' / 'test').glob('*.png'))
test_images_lowlight = list((lowlight_root / 'images' / 'test').glob('*.png'))
test_images_enhanced = list((enhanced_root / 'test').glob('*.png'))

if not test_images_enhanced:
    print("\n❌ 未找到测试图像")
    sys.exit(1)

print(f"\n找到 {len(test_images_enhanced)} 张测试图像")

# 选择要对比的图像数量
num_images_str = input("要对比多少张图像? (1-10，默认 3): ").strip()
try:
    num_images = int(num_images_str)
    num_images = max(1, min(10, num_images))
except:
    num_images = 3

print(f"\n将对比 {num_images} 张图像")

# 随机选择图像
import random
random.seed(42)
selected_images = random.sample(test_images_enhanced, min(num_images, len(test_images_enhanced)))

print("\n" + "=" * 60)
print("开始生成对比图...")
print("=" * 60)

try:
    from enlightened_gtsrb import GTSRBEnlightenGANDetector
    
    # 创建检测器
    yaml_path = Path(__file__).parent.parent / 'traffic_signs.yaml'
    detector = GTSRBEnlightenGANDetector(config_path=str(yaml_path))
    
    # 加载模型
    print("\n加载模型...")
    detector.setup_yolov8(str(model_path))
    
    # 创建输出目录
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True)
    
    # 对每张图像生成对比
    for idx, enhanced_img_path in enumerate(selected_images, 1):
        print(f"\n处理第 {idx}/{num_images} 张图像: {enhanced_img_path.name}")
        
        # 找到对应的原始和低光照图像
        img_name = enhanced_img_path.name
        
        # 尝试找到原始图像（可能文件名不完全一致）
        original_img_path = original_root / 'images' / 'test' / img_name
        lowlight_img_path = lowlight_root / 'images' / 'test' / img_name
        
        # 读取图像
        images = {}
        
        if original_img_path.exists():
            original = cv2.imread(str(original_img_path))
            images['原始图像'] = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        if lowlight_img_path.exists():
            lowlight = cv2.imread(str(lowlight_img_path))
            images['低光照图像'] = cv2.cvtColor(lowlight, cv2.COLOR_BGR2RGB)
        else:
            print(f"   ⚠️  未找到低光照图像: {lowlight_img_path.name}")
        
        enhanced = cv2.imread(str(enhanced_img_path))
        images['增强图像'] = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        # 进行检测
        print("   检测中...")
        results = detector.predict(str(enhanced_img_path), conf=0.25, save=False)
        result = results[0]
        
        annotated = result.plot()
        images['检测结果'] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # 创建对比图
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        for ax, (title, img) in zip(axes, images.items()):
            ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # 添加检测信息
        num_detections = len(result.boxes)
        fig.suptitle(f'对比 {idx}: {img_name} (检测到 {num_detections} 个标志)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_file = output_dir / f'comparison_{idx}_{img_name}'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✅ 已保存: {output_file}")
        
        plt.close()
    
    print("\n" + "=" * 60)
    print("✅ 对比图生成完成！")
    print("=" * 60)
    print(f"\n所有结果已保存到: {output_dir.absolute()}")
    
    # 显示第一张对比图
    first_comparison = list(output_dir.glob('comparison_1_*.png'))[0]
    print(f"\n正在显示第一张对比图: {first_comparison.name}")
    
    img = cv2.imread(str(first_comparison))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 5))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('对比结果示例', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n🎉 大功告成！")
    print("\n你已经完成了整个项目的流程:")
    print("  ✅ 数据准备")
    print("  ✅ 低光照模拟")
    print("  ✅ 图像增强")
    print("  ✅ 模型训练")
    print("  ✅ 模型评估")
    print("  ✅ 结果可视化")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 出现错误:")
    print("=" * 60)
    print(str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

