"""
测试 EnlightenGAN 在多张图像上的效果
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

def find_test_images(num_images=5):
    """找多张测试图像"""
    possible_dirs = [
        Path('yolo_dataset/images/train'),
        Path('yolo_dataset/images/val'),
        Path('lowlight_images/train'),
        Path('lowlight_images/val'),
    ]
    
    all_images = []
    for dir_path in possible_dirs:
        if dir_path.exists():
            for ext in ['.png', '.jpg', '.jpeg']:
                all_images.extend(list(dir_path.glob(f'*{ext}')))
    
    if not all_images:
        print("❌ 未找到图像")
        return []
    
    # 随机选择
    selected = random.sample(all_images, min(num_images, len(all_images)))
    return selected

def test_multiple_images(num_images=3):
    """测试多张图像"""
    print("\n" + "=" * 70)
    print("  EnlightenGAN 多图像测试".center(70))
    print("=" * 70)
    
    # 加载模型
    print("\n加载 EnlightenGAN 模型...")
    try:
        from enlightengan_inference import EnlightenGANInference
        model = EnlightenGANInference('weights/enlightengan.onnx')
        
        if model.session is None:
            print("❌ 模型加载失败")
            return
        
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 找测试图像
    print(f"\n寻找 {num_images} 张测试图像...")
    images = find_test_images(num_images)
    
    if not images:
        return
    
    print(f"✅ 找到 {len(images)} 张图像")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 处理每张图像
    for idx, img_path in enumerate(images, 1):
        print(f"\n处理图像 {idx}/{len(images)}: {img_path.name}")
        
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # EnlightenGAN 增强
        enhanced_gan = model.process(image)
        
        # 传统方法增强
        enhanced_trad = model.fallback_enhancement(image)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original (Low-light)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # EnlightenGAN
        axes[1].imshow(cv2.cvtColor(enhanced_gan, cv2.COLOR_BGR2RGB))
        axes[1].set_title('EnlightenGAN Enhanced', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 传统方法
        axes[2].imshow(cv2.cvtColor(enhanced_trad, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Traditional Method', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Test {idx}: {img_path.name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        output_path = f'test_enlightengan_sample_{idx}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ 保存: {output_path}")
        
        plt.close()
    
    print("\n" + "=" * 70)
    print(f"✅ 测试完成！生成了 {len(images)} 张对比图")
    print("=" * 70)
    print("\n生成的文件:")
    for i in range(1, len(images) + 1):
        print(f"   test_enlightengan_sample_{i}.png")
    
    print("\n💡 观察对比图，评估 EnlightenGAN 的效果：")
    print("   - 图像是否更清晰？")
    print("   - 细节是否保留得好？")
    print("   - 是否有过度增强或伪影？")
    print("   - 相比传统方法有明显优势吗？")

def test_specific_image(image_path):
    """测试指定图像"""
    print("\n" + "=" * 70)
    print(f"测试图像: {image_path}")
    print("=" * 70)
    
    from enlightengan_inference import EnlightenGANInference
    
    model = EnlightenGANInference('weights/enlightengan.onnx')
    if model.session is None:
        print("❌ 模型加载失败")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 无法读取图像")
        return
    
    enhanced_gan = model.process(image)
    enhanced_trad = model.fallback_enhancement(image)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(enhanced_gan, cv2.COLOR_BGR2RGB))
    axes[1].set_title('EnlightenGAN', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(enhanced_trad, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Traditional', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = 'test_enlightengan_custom.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ 对比图已保存: {output_path}")

if __name__ == '__main__':
    import sys
    
    print("\n" + "=" * 70)
    print("  EnlightenGAN 多图像测试工具".center(70))
    print("=" * 70)
    
    print("\n选择测试模式:")
    print("  1. 随机测试 3 张图像（推荐）")
    print("  2. 随机测试 5 张图像")
    print("  3. 随机测试 10 张图像")
    print("  4. 指定图像路径")
    
    choice = input("\n请选择 (1/2/3/4): ").strip()
    
    if choice == '1':
        test_multiple_images(3)
    elif choice == '2':
        test_multiple_images(5)
    elif choice == '3':
        test_multiple_images(10)
    elif choice == '4':
        img_path = input("\n输入图像路径: ").strip()
        if Path(img_path).exists():
            test_specific_image(img_path)
        else:
            print(f"❌ 文件不存在: {img_path}")
    else:
        print("❌ 无效选择，默认测试 3 张")
        test_multiple_images(3)

