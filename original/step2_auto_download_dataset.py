"""
步骤 2 (自动版): 自动下载 GTSRB 数据集
使用 kagglehub 自动从 Kaggle 下载数据集
"""

import sys
from pathlib import Path

print("=" * 60)
print("📥 步骤 2: 自动下载 GTSRB 数据集")
print("=" * 60)

# 检查 kagglehub 是否安装
print("\n检查 kagglehub...")
try:
    import kagglehub
    print("✅ kagglehub 已安装")
except ImportError:
    print("❌ kagglehub 未安装")
    print("\n正在安装 kagglehub...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub
        print("✅ kagglehub 安装成功")
    except Exception as e:
        print(f"❌ 安装失败: {e}")
        print("\n请手动安装: pip install kagglehub")
        sys.exit(1)

# 说明
print("\n" + "=" * 60)
print("📚 关于 Kaggle 数据集下载:")
print("=" * 60)
print("""
kagglehub 会自动下载数据集到本地缓存目录。

优点:
✅ 不需要手动下载 zip 文件
✅ 不需要翻墙（大部分情况）
✅ 自动解压和管理
✅ 支持断点续传

注意:
⚠️ 第一次下载可能需要 Kaggle 账号验证
⚠️ 数据集大小约 300-500 MB，需要一些时间
⚠️ 会下载到系统缓存目录，然后我们会复制到项目目录
""")

# 确认
response = input("\n是否开始下载? (输入 yes 继续): ").strip().lower()

if response != 'yes':
    print("\n❌ 用户取消下载")
    print("\n如果你已经有数据集，可以运行:")
    print("   python step2_download_dataset_guide.py")
    sys.exit(0)

# 开始下载
print("\n" + "=" * 60)
print("开始下载数据集...")
print("=" * 60)
print("\n⏳ 这可能需要几分钟，请耐心等待...")
print("   (如果第一次使用，可能会提示你配置 Kaggle API)")

try:
    # 下载数据集
    print("\n正在从 Kaggle 下载...")
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    
    print("\n✅ 下载完成！")
    print(f"数据集路径: {path}")
    
    # 检查下载的内容
    dataset_path = Path(path)
    
    if not dataset_path.exists():
        print(f"\n❌ 错误: 路径不存在: {dataset_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("检查数据集结构...")
    print("=" * 60)
    
    # 列出下载的内容
    print("\n下载的文件和目录:")
    for item in dataset_path.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.1f} MB)")
    
    # 查找训练和测试目录
    train_found = False
    test_found = False
    
    # 常见的目录结构
    possible_train_paths = [
        dataset_path / 'Final_Training' / 'Images',
        dataset_path / 'Train',
        dataset_path / 'train',
    ]
    
    possible_test_paths = [
        dataset_path / 'Final_Test' / 'Images',
        dataset_path / 'Test',
        dataset_path / 'test',
    ]
    
    train_path = None
    test_path = None
    
    for p in possible_train_paths:
        if p.exists():
            train_path = p
            train_found = True
            break
    
    for p in possible_test_paths:
        if p.exists():
            test_path = p
            test_found = True
            break
    
    print("\n" + "=" * 60)
    print("验证数据集:")
    print("=" * 60)
    
    if train_found and train_path:
        num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
        print(f"✅ 训练集目录: {train_path}")
        print(f"   包含 {num_classes} 个类别")
    else:
        print("⚠️  未找到标准的训练集目录")
        print("   数据集结构可能不同，需要手动检查")
    
    if test_found and test_path:
        num_test = len(list(test_path.glob('*.ppm'))) + len(list(test_path.glob('*.jpg')))
        print(f"✅ 测试集目录: {test_path}")
        print(f"   包含 {num_test} 张图片")
    else:
        print("⚠️  未找到标准的测试集目录")
    
    # 保存路径
    config_file = Path(__file__).parent / 'dataset_path.txt'
    with open(config_file, 'w') as f:
        f.write(str(dataset_path.absolute()))
    
    print("\n" + "=" * 60)
    print("✅ 数据集下载和验证完成！")
    print("=" * 60)
    print(f"\n数据集位置: {dataset_path}")
    print(f"配置已保存: {config_file}")
    
    print("\n" + "=" * 60)
    print("📝 下一步:")
    print("=" * 60)
    print("运行以下命令开始转换数据集格式:")
    print("   python step3_convert_dataset.py")
    print("\n转换脚本会自动读取刚才保存的路径。")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 下载过程中出现错误:")
    print("=" * 60)
    print(str(e))
    
    print("\n可能的原因:")
    print("1. 需要配置 Kaggle API")
    print("2. 网络连接问题")
    print("3. 磁盘空间不足")
    
    print("\n" + "=" * 60)
    print("Kaggle API 配置方法:")
    print("=" * 60)
    print("""
如果提示需要 Kaggle API Token:

1. 访问 Kaggle 网站: https://www.kaggle.com/
2. 登录你的账号
3. 点击右上角头像 → Settings
4. 滚动到 "API" 部分
5. 点击 "Create New API Token"
6. 会下载一个 kaggle.json 文件

Windows 用户:
- 将 kaggle.json 放到: C:\\Users\\你的用户名\\.kaggle\\

Linux/Mac 用户:
- 将 kaggle.json 放到: ~/.kaggle/
- 运行: chmod 600 ~/.kaggle/kaggle.json

然后重新运行此脚本。
    """)
    
    print("\n或者使用手动下载方式:")
    print("   python step2_download_dataset_guide.py")
    
    sys.exit(1)

