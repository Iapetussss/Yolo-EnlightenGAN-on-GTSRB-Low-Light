"""
步骤 1: 检查环境配置
这个脚本会检查你的 Python 环境是否正确配置
"""

import sys

print("=" * 60)
print("🔍 步骤 1: 检查环境配置")
print("=" * 60)

# 检查 Python 版本
print("\n1. Python 版本检查...")
python_version = sys.version_info
print(f"   当前 Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major >= 3 and python_version.minor >= 8:
    print("   ✅ Python 版本符合要求 (>= 3.8)")
else:
    print("   ❌ Python 版本过低，请升级到 3.8 或更高版本")
    sys.exit(1)

# 检查必要的包
print("\n2. 检查必要的包...")

packages_to_check = {
    'torch': 'PyTorch',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'tqdm': 'tqdm',
    'matplotlib': 'Matplotlib'
}

missing_packages = []

for package, name in packages_to_check.items():
    try:
        if package == 'cv2':
            import cv2
        elif package == 'PIL':
            from PIL import Image
        else:
            __import__(package)
        print(f"   ✅ {name} 已安装")
    except ImportError:
        print(f"   ❌ {name} 未安装")
        missing_packages.append(name)

if missing_packages:
    print(f"\n⚠️  缺少以下包: {', '.join(missing_packages)}")
    print("   请运行以下命令安装：")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# 检查 Ultralytics (YOLOv8)
print("\n3. 检查 YOLOv8 (Ultralytics)...")
try:
    from ultralytics import YOLO
    print("   ✅ Ultralytics (YOLOv8) 已安装")
except ImportError:
    print("   ❌ Ultralytics 未安装")
    print("   请运行: pip install ultralytics")
    sys.exit(1)

# 检查 GPU
print("\n4. 检查 GPU 可用性...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ GPU 可用")
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
    else:
        print("   ⚠️  未检测到 GPU，将使用 CPU 训练")
        print("   （CPU 训练会比较慢，但也可以完成）")
except:
    print("   ⚠️  无法检查 GPU 状态")

# 检查目录结构
print("\n5. 检查项目目录结构...")
from pathlib import Path

project_root = Path(__file__).parent
required_files = [
    'enlightened_gtsrb.py',
    'data_preparation.py',
    'enlightengan_inference.py',
    'requirements.txt'
]

all_exist = True
for file in required_files:
    file_path = project_root / file
    if file_path.exists():
        print(f"   ✅ {file} 存在")
    else:
        print(f"   ❌ {file} 不存在")
        all_exist = False

if not all_exist:
    print("\n   ⚠️  部分文件缺失，请检查项目完整性")

# 检查配置文件
print("\n6. 检查配置文件...")
yaml_path = project_root.parent / 'traffic_signs.yaml'
if yaml_path.exists():
    print(f"   ✅ traffic_signs.yaml 存在")
else:
    print(f"   ❌ traffic_signs.yaml 不存在")
    print(f"   期望路径: {yaml_path}")

# 总结
print("\n" + "=" * 60)
if not missing_packages and all_exist:
    print("🎉 环境检查完成！所有必要组件都已就绪。")
    print("\n下一步: 运行 step2_download_dataset_guide.py")
else:
    print("⚠️  环境配置不完整，请按照上述提示解决问题。")
print("=" * 60)

