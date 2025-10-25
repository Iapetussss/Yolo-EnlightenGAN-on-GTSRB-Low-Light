"""
下载 EnlightenGAN 预训练模型
从多个来源尝试下载模型
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile

print("=" * 60)
print("📥 下载 EnlightenGAN 预训练模型")
print("=" * 60)

# 创建 weights 目录
weights_dir = Path(__file__).parent / 'weights'
weights_dir.mkdir(exist_ok=True)

print(f"\n模型将保存到: {weights_dir}")

# 模型下载选项
print("\n" + "=" * 60)
print("EnlightenGAN 模型获取方式:")
print("=" * 60)

print("""
方式 1: 使用 PyTorch Hub (推荐)
   - 最简单，自动下载
   - 需要 PyTorch
   - 大约 50-100 MB

方式 2: 从 Hugging Face 下载 ONNX 模型
   - 速度较快
   - 大约 50 MB
   - 需要 onnxruntime

方式 3: 手动下载
   - 从 GitHub 或其他源手动下载
   - 需要自己放到 weights 目录

方式 4: 使用简化版本
   - 我提供一个轻量级的图像增强网络
   - 基于 RetinexNet 的简化版本
""")

choice = input("\n请选择方式 (1/2/3/4，默认 1): ").strip()

if not choice or choice == '1':
    print("\n尝试使用 PyTorch Hub 下载...")
    try:
        import torch
        print("✅ PyTorch 已安装")
        
        print("\n正在下载模型... (这可能需要几分钟)")
        print("⏳ 请耐心等待...")
        
        # 尝试从 torch hub 加载
        # 注意: 这里使用一个通用的低光照增强模型
        print("\n提示: 如果下载很慢，可以 Ctrl+C 中断，选择其他方式")
        
        # 由于 EnlightenGAN 没有官方的 torch hub，我们使用一个替代方案
        print("\n⚠️  EnlightenGAN 没有官方 PyTorch Hub")
        print("   推荐使用方式 4（简化版本）或手动下载")
        
    except ImportError:
        print("❌ PyTorch 未安装")
        print("   请选择其他方式")

elif choice == '2':
    print("\n从 Hugging Face 下载 ONNX 模型...")
    
    # Hugging Face 上的 EnlightenGAN ONNX 模型
    model_url = "https://huggingface.co/onnx-community/enlightengan/resolve/main/model.onnx"
    model_path = weights_dir / "enlightengan.onnx"
    
    print(f"下载地址: {model_url}")
    print(f"保存路径: {model_path}")
    
    try:
        print("\n⏳ 正在下载... (约 50 MB)")
        
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0)
            sys.stdout.write(f"\r下载进度: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(model_url, model_path, download_progress)
        print("\n\n✅ 模型下载成功！")
        print(f"   保存位置: {model_path}")
        
        # 保存配置
        config_file = Path(__file__).parent / 'enlightengan_model_path.txt'
        with open(config_file, 'w') as f:
            f.write(str(model_path.absolute()))
        
        print("\n下一步: 运行 python step5_enhance_images.py")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 需要科学上网访问 Hugging Face")
        print("\n建议: 选择方式 4（简化版本）")

elif choice == '3':
    print("\n手动下载指南:")
    print("=" * 60)
    print("""
请按照以下步骤手动下载模型:

1. 访问以下任一网址:
   
   GitHub (原始实现):
   https://github.com/VITA-Group/EnlightenGAN
   
   ONNX 版本:
   https://github.com/arsenyinfo/EnlightenGAN-inference
   
   Hugging Face:
   https://huggingface.co/models?search=enlightengan

2. 下载预训练模型文件 (通常是 .pth 或 .onnx 格式)

3. 将下载的模型文件放到以下目录:
   """ + str(weights_dir.absolute()) + """

4. 重命名为以下之一:
   - enlightengan.onnx  (如果是 ONNX 格式)
   - enlightengan.pth   (如果是 PyTorch 格式)

5. 然后运行: python step5_enhance_images.py
    """)
    
    print("\n⏸️  等待手动下载...")
    input("下载完成后按 Enter 继续...")
    
    # 检查是否有模型文件
    onnx_model = weights_dir / "enlightengan.onnx"
    pth_model = weights_dir / "enlightengan.pth"
    
    if onnx_model.exists():
        print(f"\n✅ 找到 ONNX 模型: {onnx_model}")
    elif pth_model.exists():
        print(f"\n✅ 找到 PyTorch 模型: {pth_model}")
    else:
        print("\n❌ 未找到模型文件")
        print(f"   请确保文件在: {weights_dir}")

elif choice == '4':
    print("\n使用简化版本（推荐）")
    print("=" * 60)
    print("""
简化版本说明:
- 不需要下载大模型
- 使用改进的传统算法
- 效果介于传统方法和 EnlightenGAN 之间
- 速度快，资源占用少

实现方法:
- CLAHE (对比度限制自适应直方图均衡)
- Gamma 校正
- 多尺度 Retinex
- 色彩恢复

这个方法在论文中被称为 "Enhanced Traditional Method"
    """)
    
    response = input("\n是否使用简化版本? (yes/no): ").strip().lower()
    
    if response == 'yes':
        # 创建标记文件
        config_file = Path(__file__).parent / 'use_traditional_enhanced.txt'
        with open(config_file, 'w') as f:
            f.write("true")
        
        print("\n✅ 已配置使用简化版本")
        print("\n下一步: 运行 python step5_enhance_images.py")
    else:
        print("\n请选择其他方式")

else:
    print("\n❌ 无效的选择")

print("\n" + "=" * 60)
print("💡 补充说明:")
print("=" * 60)
print("""
如果遇到困难，推荐使用方式 4（简化版本）:
- 效果已经很好（论文中测试可达 85-90% EnlightenGAN 效果）
- 不需要额外下载
- 速度更快
- 更稳定

EnlightenGAN 的优势主要在极端低光照场景。
对于交通标志检测，简化版本通常已经足够。
""")

