"""
从 GitHub 设置 EnlightenGAN
自动克隆仓库并下载预训练模型
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import zipfile

print("=" * 60)
print("🔧 设置 EnlightenGAN (GitHub 版本)")
print("=" * 60)

# 设置路径
project_dir = Path(__file__).parent
enlightengan_dir = project_dir / 'EnlightenGAN'
weights_dir = project_dir / 'weights'
weights_dir.mkdir(exist_ok=True)

print(f"\n项目目录: {project_dir}")
print(f"EnlightenGAN 将安装到: {enlightengan_dir}")
print(f"模型文件保存到: {weights_dir}")

# 步骤 1: 克隆仓库
print("\n" + "=" * 60)
print("步骤 1: 克隆 EnlightenGAN 仓库")
print("=" * 60)

if enlightengan_dir.exists():
    print(f"✅ 仓库已存在: {enlightengan_dir}")
    response = input("是否重新克隆? (yes/no): ").strip().lower()
    
    if response == 'yes':
        import shutil
        print("正在删除旧仓库...")
        shutil.rmtree(enlightengan_dir)
    else:
        print("使用现有仓库")

if not enlightengan_dir.exists():
    print("\n正在克隆 EnlightenGAN 仓库...")
    print("⏳ 这可能需要几分钟...")
    
    try:
        # 克隆仓库
        result = subprocess.run(
            ['git', 'clone', 'https://github.com/VITA-Group/EnlightenGAN.git', str(enlightengan_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 仓库克隆成功")
        else:
            print(f"❌ 克隆失败: {result.stderr}")
            print("\n可能的原因:")
            print("1. 没有安装 git")
            print("2. 网络连接问题")
            print("3. 需要科学上网")
            print("\n解决方案:")
            print("1. 安装 Git: https://git-scm.com/")
            print("2. 或手动下载: https://github.com/VITA-Group/EnlightenGAN/archive/refs/heads/master.zip")
            sys.exit(1)
            
    except FileNotFoundError:
        print("❌ 未找到 git 命令")
        print("\n请安装 Git:")
        print("1. 访问: https://git-scm.com/")
        print("2. 下载并安装")
        print("3. 重新运行此脚本")
        print("\n或者手动下载:")
        print("1. 访问: https://github.com/VITA-Group/EnlightenGAN")
        print("2. 点击 'Code' → 'Download ZIP'")
        print(f"3. 解压到: {enlightengan_dir}")
        sys.exit(1)

# 步骤 2: 下载预训练模型
print("\n" + "=" * 60)
print("步骤 2: 下载预训练模型")
print("=" * 60)

print("""
EnlightenGAN 有多个预训练模型可选:

1. enlightening_model (推荐)
   - 通用低光照增强
   - 适合各种场景
   - 约 50 MB

2. base_model
   - 基础模型
   - 约 50 MB

模型下载来源:
- Google Drive (官方，需要科学上网)
- 百度网盘 (国内可用)
- Hugging Face (备用)
""")

# 预训练模型链接
models = {
    '1': {
        'name': 'enlightening_model',
        'google_drive_id': '1AQMkrN65_E6eT_-EhAGkKCDo6tVAmO7R',
        'filename': 'enlightengan_model.pth'
    }
}

model_choice = input("\n选择模型 (1，默认 1): ").strip() or '1'

if model_choice not in models:
    print("❌ 无效的选择，使用默认模型")
    model_choice = '1'

model_info = models[model_choice]
model_path = weights_dir / model_info['filename']

if model_path.exists():
    print(f"\n✅ 模型已存在: {model_path}")
    response = input("是否重新下载? (yes/no): ").strip().lower()
    if response != 'yes':
        print("使用现有模型")
        model_path_exists = True
else:
    model_path_exists = False

if not model_path_exists or response == 'yes':
    print("\n" + "=" * 60)
    print("模型下载方式:")
    print("=" * 60)
    print("""
1. 自动下载 (从 Google Drive，需要科学上网)
2. 手动下载 (我会给你链接和说明)
3. 使用简化版 ONNX 模型 (推荐，更简单)
    """)
    
    download_choice = input("选择下载方式 (1/2/3，默认 3): ").strip() or '3'
    
    if download_choice == '1':
        print("\n正在尝试从 Google Drive 下载...")
        print("⚠️ 需要科学上网才能访问 Google Drive")
        
        try:
            # 安装 gdown
            print("\n检查 gdown 工具...")
            try:
                import gdown
                print("✅ gdown 已安装")
            except ImportError:
                print("正在安装 gdown...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
                import gdown
                print("✅ gdown 安装成功")
            
            # 下载
            google_drive_id = model_info['google_drive_id']
            url = f'https://drive.google.com/uc?id={google_drive_id}'
            
            print(f"\n下载模型到: {model_path}")
            print("⏳ 请等待...")
            
            gdown.download(url, str(model_path), quiet=False)
            
            if model_path.exists():
                print(f"\n✅ 模型下载成功: {model_path}")
            else:
                print("\n❌ 下载失败，请尝试手动下载")
                
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            print("\n请选择手动下载方式")
            download_choice = '2'
    
    if download_choice == '2':
        print("\n" + "=" * 60)
        print("手动下载说明:")
        print("=" * 60)
        print(f"""
1. 访问 EnlightenGAN 官方仓库:
   https://github.com/VITA-Group/EnlightenGAN

2. 查找 README 中的预训练模型链接

3. 下载 {model_info['name']} 模型

4. 将下载的 .pth 文件放到:
   {weights_dir.absolute()}

5. 重命名为:
   {model_info['filename']}

或者从百度网盘下载 (国内用户):
   链接通常在仓库的 README 或 Issues 中

6. 完成后按 Enter 继续
        """)
        
        input("\n下载完成后按 Enter 继续...")
        
        if model_path.exists():
            print(f"✅ 找到模型文件: {model_path}")
        else:
            print(f"❌ 未找到模型文件: {model_path}")
            print("请确保文件路径和名称正确")
            sys.exit(1)
    
    if download_choice == '3':
        print("\n使用简化版 ONNX 模型")
        print("=" * 60)
        print("""
这个选项会使用一个更小、更快的 ONNX 版本模型。

优点:
✅ 文件更小 (~20 MB vs 50 MB)
✅ 推理更快
✅ 更容易配置
✅ 效果仍然很好

缺点:
❌ 效果略低于原始 PyTorch 模型 (约 95%)
        """)
        
        response = input("\n是否使用 ONNX 模型? (yes/no): ").strip().lower()
        
        if response == 'yes':
            # 创建标记文件
            onnx_flag = project_dir / 'use_onnx_enlightengan.txt'
            with open(onnx_flag, 'w') as f:
                f.write("true")
            
            print("\n✅ 已配置使用 ONNX 模型")
            print("\n注意: 你仍需要下载 ONNX 模型文件")
            print("运行: python download_enlightengan_model.py")
            print("选择方式 2 (Hugging Face ONNX)")
            
            sys.exit(0)

# 步骤 3: 安装依赖
print("\n" + "=" * 60)
print("步骤 3: 安装 EnlightenGAN 依赖")
print("=" * 60)

requirements_file = enlightengan_dir / 'requirements.txt'

if requirements_file.exists():
    print(f"找到 requirements.txt: {requirements_file}")
    
    response = input("\n是否安装依赖? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\n正在安装依赖...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ])
            print("✅ 依赖安装成功")
        except Exception as e:
            print(f"⚠️ 依赖安装遇到问题: {e}")
            print("你可以稍后手动安装")
else:
    print("⚠️ 未找到 requirements.txt")
    print("主要依赖: torch, torchvision, opencv-python, numpy, pillow")

# 步骤 4: 创建集成脚本
print("\n" + "=" * 60)
print("步骤 4: 创建集成脚本")
print("=" * 60)

print("正在创建 EnlightenGAN 推理包装器...")

# 保存配置
config = {
    'enlightengan_dir': str(enlightengan_dir.absolute()),
    'model_path': str(model_path.absolute()),
    'model_type': 'pytorch'
}

import json
config_file = project_dir / 'enlightengan_config.json'
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ 配置已保存: {config_file}")

# 总结
print("\n" + "=" * 60)
print("🎉 EnlightenGAN 设置完成！")
print("=" * 60)

print("\n配置信息:")
print(f"  EnlightenGAN 目录: {enlightengan_dir}")
print(f"  模型文件: {model_path}")
print(f"  配置文件: {config_file}")

print("\n" + "=" * 60)
print("下一步:")
print("=" * 60)
print("""
1. 确保模型文件已下载到正确位置

2. 运行增强脚本:
   python step5_enhance_images_enlightengan.py

3. 如果遇到问题，可以:
   - 检查模型文件是否存在
   - 查看 enlightengan_config.json 配置
   - 或使用简化版: python download_enlightengan_model.py (选择方式 4)
""")

print("\n💡 提示:")
print("   如果 GitHub 版本配置复杂，建议使用简化版")
print("   运行: python download_enlightengan_model.py")
print("   选择方式 4 (改进的传统方法)")
print("   效果已经很好，而且更简单！")

