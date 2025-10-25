"""
测试图像增强效果
对比原始、低光照、增强后的图像
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

print("=" * 60)
print("🖼️  测试图像增强效果")
print("=" * 60)

# 查找数据集路径
project_dir = Path(__file__).parent

original_dir = project_dir / 'traffic_sign_data' / 'original' / 'images' / 'train'
lowlight_dir = project_dir / 'traffic_sign_data' / 'low_light' / 'images' / 'train'
enhanced_dir = project_dir / 'traffic_sign_data' / 'enhanced_images' / 'train'

# 检查目录是否存在
if not enhanced_dir.exists():
    print(f"❌ 增强图像目录不存在: {enhanced_dir}")
    print("   请先运行 step5_enhance_images.py")
    exit(1)

print(f"\n✅ 原始图像: {original_dir}")
print(f"✅ 低光照图像: {lowlight_dir}")
print(f"✅ 增强图像: {enhanced_dir}")

# 获取图像列表
enhanced_images = list(enhanced_dir.glob('*.png'))

if not enhanced_images:
    print("\n❌ 未找到增强图像")
    exit(1)

print(f"\n找到 {len(enhanced_images)} 张增强图像")

# 随机选择几张图像进行对比
num_samples = 5
samples = random.sample(enhanced_images, min(num_samples, len(enhanced_images)))

print(f"随机选择 {len(samples)} 张图像进行对比\n")

# 创建对比图
for idx, enhanced_path in enumerate(samples, 1):
    print(f"处理第 {idx}/{len(samples)} 张图像: {enhanced_path.name}")
    
    # 构建对应的原始和低光照图像路径
    img_name = enhanced_path.name
    original_path = original_dir / img_name
    lowlight_path = lowlight_dir / img_name
    
    # 读取图像
    images = {}
    titles = []
    
    if original_path.exists():
        original = cv2.imread(str(original_path))
        images['Original'] = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    if lowlight_path.exists():
        lowlight = cv2.imread(str(lowlight_path))
        images['Low-light'] = cv2.cvtColor(lowlight, cv2.COLOR_BGR2RGB)
    
    enhanced = cv2.imread(str(enhanced_path))
    images['Enhanced'] = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    # 显示对比
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for ax, (title, img) in zip(axes, images.items()):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(f'Comparison {idx}: {img_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存对比图
    output_dir = project_dir / 'enhancement_comparison'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'comparison_{idx}_{img_name}'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存到: {output_file}")
    
    # 显示第一张
    if idx == 1:
        print(f"\n  正在显示第一张对比图...")
        plt.show()
    else:
        plt.close()

print("\n" + "=" * 60)
print("✅ 对比图生成完成！")
print("=" * 60)

print(f"\n所有对比图已保存到: {output_dir.absolute()}")

print("\n" + "=" * 60)
print("📊 图像增强效果评估:")
print("=" * 60)

# 计算平均亮度提升
print("\n计算平均亮度变化...")

brightness_improvements = []

for enhanced_path in random.sample(enhanced_images, min(100, len(enhanced_images))):
    img_name = enhanced_path.name
    lowlight_path = lowlight_dir / img_name
    
    if lowlight_path.exists():
        lowlight = cv2.imread(str(lowlight_path), cv2.IMREAD_GRAYSCALE)
        enhanced = cv2.imread(str(enhanced_path), cv2.IMREAD_GRAYSCALE)
        
        lowlight_brightness = np.mean(lowlight)
        enhanced_brightness = np.mean(enhanced)
        
        improvement = (enhanced_brightness - lowlight_brightness) / lowlight_brightness * 100
        brightness_improvements.append(improvement)

if brightness_improvements:
    avg_improvement = np.mean(brightness_improvements)
    print(f"\n平均亮度提升: {avg_improvement:.1f}%")
    print(f"亮度提升范围: {min(brightness_improvements):.1f}% ~ {max(brightness_improvements):.1f}%")
else:
    print("\n无法计算亮度提升")

print("\n" + "=" * 60)
print("💡 评估结论:")
print("=" * 60)

if brightness_improvements and avg_improvement > 0:
    if avg_improvement > 50:
        print("✅ 优秀！图像亮度提升明显")
    elif avg_improvement > 30:
        print("✅ 良好！图像增强效果不错")
    elif avg_improvement > 10:
        print("⚠️  一般。增强效果较弱")
    else:
        print("❌ 较差。可能需要调整增强参数")
    
    print(f"\n图像已成功增强，平均亮度提升了 {avg_improvement:.1f}%")
    print("可以继续进行模型训练！")
else:
    print("⚠️  无法评估增强效果")

print("\n" + "=" * 60)
print("下一步:")
print("=" * 60)
print("运行以下命令开始训练模型:")
print("   python step6_train_model.py")
print("=" * 60)

