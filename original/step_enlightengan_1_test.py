"""
步骤 1：测试 EnlightenGAN 模型
- 检查模型文件是否存在
- 加载模型
- 测试单张图像增强
- 生成对比图
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

def check_model_file():
    """检查模型文件"""
    print("\n" + "=" * 60)
    print("检查模型文件...")
    print("=" * 60)
    
    model_path = Path('weights/enlightengan.onnx')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ 找到模型: {model_path}")
        print(f"   文件大小: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        print("\n请先下载 EnlightenGAN 模型！")
        print("\n方法 1：从 GitHub 下载")
        print("  访问: https://github.com/arsenyinfo/EnlightenGAN-inference")
        print("  下载 enlightengan.onnx 并放到 weights/ 目录")
        print("\n方法 2：从 Google Drive 下载")
        print("  访问: https://drive.google.com/drive/folders/1i_Y6c3vl3iZpNJFcjB5FW1LRVmYSKMqF")
        print("  下载模型文件")
        print("\n方法 3：运行自动下载脚本（如果网络允许）")
        print("  python download_enlightengan_onnx.py")
        print("\n方法 4：使用传统增强方法（当前使用的方法）")
        print("  你的传统方法已达到 98.65% mAP，效果已经很好！")
        return False

def load_enlightengan():
    """加载 EnlightenGAN 模型"""
    print("\n" + "=" * 60)
    print("加载 EnlightenGAN 模型...")
    print("=" * 60)
    
    try:
        from enlightengan_inference import EnlightenGANInference
        
        model = EnlightenGANInference('weights/enlightengan.onnx')
        
        if model.session is not None:
            print("✅ EnlightenGAN 模型加载成功！")
            print(f"   使用设备: {model.session.get_providers()[0]}")
            return model
        else:
            print("❌ 模型加载失败")
            return None
            
    except Exception as e:
        print(f"❌ 加载模型出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_test_image():
    """找一张测试图像"""
    print("\n" + "=" * 60)
    print("寻找测试图像...")
    print("=" * 60)
    
    # 尝试从低光照数据集中找一张图像
    possible_dirs = [
        'lowlight_images/train',
        'lowlight_images/val',
        'traffic_sign_data/lowlight_images/train',
        'yolo_dataset/images/train'
    ]
    
    for dir_path in possible_dirs:
        dir_p = Path(dir_path)
        if dir_p.exists():
            # 找第一张图像
            for ext in ['.png', '.jpg', '.jpeg', '.ppm']:
                images = list(dir_p.glob(f'*{ext}'))
                if images:
                    print(f"✅ 找到测试图像: {images[0]}")
                    return images[0]
    
    print("❌ 未找到测试图像")
    print("   请确保你已经完成了数据准备步骤")
    return None

def test_enhancement(model, test_image_path):
    """测试图像增强"""
    print("\n" + "=" * 60)
    print("测试图像增强...")
    print("=" * 60)
    
    try:
        # 读取图像
        image = cv2.imread(str(test_image_path))
        if image is None:
            print(f"❌ 无法读取图像: {test_image_path}")
            return None
        
        print(f"   图像尺寸: {image.shape}")
        
        # EnlightenGAN 增强
        import time
        start = time.time()
        enhanced_gan = model.process(image)
        elapsed = (time.time() - start) * 1000
        
        print(f"✅ EnlightenGAN 增强成功！")
        print(f"   处理时间: {elapsed:.1f} ms")
        
        # 传统方法增强（对比）
        enhanced_traditional = model.fallback_enhancement(image)
        
        return {
            'original': image,
            'enlightengan': enhanced_gan,
            'traditional': enhanced_traditional
        }
        
    except Exception as e:
        print(f"❌ 增强失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_comparison(images, output_path='test_enlightengan_comparison.png'):
    """创建对比图"""
    print("\n" + "=" * 60)
    print("生成对比图...")
    print("=" * 60)
    
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始图像
        axes[0].imshow(cv2.cvtColor(images['original'], cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original (Low-light)', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # EnlightenGAN 增强
        axes[1].imshow(cv2.cvtColor(images['enlightengan'], cv2.COLOR_BGR2RGB))
        axes[1].set_title('EnlightenGAN Enhanced', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        # 传统方法增强
        axes[2].imshow(cv2.cvtColor(images['traditional'], cv2.COLOR_BGR2RGB))
        axes[2].set_title('Traditional Method (CLAHE+Gamma)', fontsize=16, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle('Enhancement Method Comparison', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 对比图已保存: {output_path}")
        
        # 显示
        plt.show()
        
        return output_path
        
    except Exception as e:
        print(f"❌ 生成对比图失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_metrics(images):
    """计算图像质量指标"""
    print("\n" + "=" * 60)
    print("计算图像质量指标...")
    print("=" * 60)
    
    def get_brightness(img):
        """计算平均亮度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def get_contrast(img):
        """计算对比度（标准差）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    metrics = {}
    for name, img in images.items():
        brightness = get_brightness(img)
        contrast = get_contrast(img)
        metrics[name] = {
            'brightness': brightness,
            'contrast': contrast
        }
    
    print("\n指标对比:")
    print(f"{'方法':<20} {'亮度':<15} {'对比度':<15}")
    print("-" * 50)
    for name, vals in metrics.items():
        print(f"{name:<20} {vals['brightness']:<15.2f} {vals['contrast']:<15.2f}")
    
    return metrics

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  EnlightenGAN 测试程序".center(70))
    print("=" * 70)
    
    # 1. 检查模型文件
    if not check_model_file():
        print("\n❌ 测试终止：缺少模型文件")
        sys.exit(1)
    
    # 2. 加载模型
    model = load_enlightengan()
    if model is None:
        print("\n❌ 测试终止：模型加载失败")
        sys.exit(1)
    
    # 3. 找测试图像
    test_image = find_test_image()
    if test_image is None:
        print("\n❌ 测试终止：未找到测试图像")
        print("\n提示：如果还没有数据，可以先运行:")
        print("  python step4_create_lowlight.py")
        sys.exit(1)
    
    # 4. 测试增强
    images = test_enhancement(model, test_image)
    if images is None:
        print("\n❌ 测试终止：增强失败")
        sys.exit(1)
    
    # 5. 计算指标
    metrics = calculate_metrics(images)
    
    # 6. 生成对比图
    comparison_path = create_comparison(images)
    if comparison_path is None:
        print("\n⚠️  警告：对比图生成失败，但增强功能正常")
    
    # 7. 总结
    print("\n" + "=" * 70)
    print("  测试完成！".center(70))
    print("=" * 70)
    
    print("\n✅ EnlightenGAN 工作正常！")
    print(f"\n📊 结果分析:")
    print(f"   - EnlightenGAN 亮度: {metrics['enlightengan']['brightness']:.1f}")
    print(f"   - 传统方法亮度:     {metrics['traditional']['brightness']:.1f}")
    print(f"   - 原图亮度:         {metrics['original']['brightness']:.1f}")
    
    print(f"\n📈 对比图已保存: {comparison_path}")
    print(f"   打开查看效果，决定是否使用 EnlightenGAN")
    
    print("\n🎯 下一步:")
    print("   如果 EnlightenGAN 效果好 → 运行:")
    print("     python step_enlightengan_2_enhance_dataset.py")
    print("\n   如果效果一般 → 继续使用传统方法（当前 98.65% mAP 已经很好）")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    main()

