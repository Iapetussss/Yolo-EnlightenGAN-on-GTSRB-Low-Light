"""
步骤 2: 数据集下载指南
这个脚本会指导你如何下载 GTSRB 数据集
"""

print("=" * 60)
print("📥 步骤 2: 下载 GTSRB 数据集")
print("=" * 60)

print("""
GTSRB (German Traffic Sign Recognition Benchmark) 是一个
德国交通标志识别基准数据集，包含 43 类交通标志。

📊 数据集统计:
- 训练图片: ~39,000 张
- 测试图片: ~12,600 张
- 类别数: 43 类
- 图片格式: PPM (需要转换)

""")

print("🔗 下载方式 1: Kaggle (推荐)")
print("-" * 60)
print("""
1. 访问 Kaggle 网站:
   https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

2. 如果没有 Kaggle 账号，需要先注册（免费）

3. 点击页面右上角的 "Download" 按钮

4. 下载完成后，会得到一个 zip 文件 (约 300-500 MB)

5. 解压到一个目录，例如:
   D:\\datasets\\GTSRB\\

6. 解压后应该看到以下结构:
   GTSRB/
   ├── Final_Training/
   │   └── Images/
   │       ├── 00000/
   │       ├── 00001/
   │       └── ...
   └── Final_Test/
       └── Images/
           ├── 00000.ppm
           ├── 00001.ppm
           └── GT-final_test.csv
""")

print("\n🔗 下载方式 2: 官方网站")
print("-" * 60)
print("""
1. 访问官方网站:
   https://benchmark.ini.rub.de/gtsrb_dataset.html

2. 下载以下两个文件:
   - GTSRB_Final_Training_Images.zip (训练集)
   - GTSRB_Final_Test_Images.zip (测试集)
   - GTSRB_Final_Test_GT.zip (测试集标注)

3. 解压到同一个目录
""")

print("\n🔗 下载方式 3: 使用 Python 脚本自动下载 (高级)")
print("-" * 60)
print("""
如果你熟悉 Python，可以使用 Kaggle API:

1. 安装 Kaggle API:
   pip install kaggle

2. 配置 Kaggle 凭证 (需要从 Kaggle 网站获取)

3. 运行下载命令:
   kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

4. 解压文件:
   unzip gtsrb-german-traffic-sign.zip -d D:/datasets/GTSRB/
""")

print("\n" + "=" * 60)
print("✅ 下载完成后的检查:")
print("=" * 60)

from pathlib import Path

# 让用户输入数据集路径
print("\n请输入你下载并解压后的 GTSRB 数据集路径:")
print("例如: D:\\datasets\\GTSRB 或 D:/datasets/GTSRB")
print("(如果还没下载，可以直接按 Enter 跳过)")

dataset_path = input("\n数据集路径: ").strip()

if dataset_path:
    dataset_path = Path(dataset_path)
    
    if dataset_path.exists():
        print("\n✅ 路径存在！正在检查目录结构...")
        
        # 检查训练集
        train_path = dataset_path / 'Final_Training' / 'Images'
        if train_path.exists():
            num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
            print(f"   ✅ 训练集目录存在，包含 {num_classes} 个类别")
        else:
            print(f"   ❌ 训练集目录不存在: {train_path}")
        
        # 检查测试集
        test_path = dataset_path / 'Final_Test' / 'Images'
        if test_path.exists():
            num_test_images = len(list(test_path.glob('*.ppm')))
            print(f"   ✅ 测试集目录存在，包含 {num_test_images} 张图片")
        else:
            print(f"   ❌ 测试集目录不存在: {test_path}")
        
        # 保存路径供下一步使用
        config_file = Path(__file__).parent / 'dataset_path.txt'
        with open(config_file, 'w') as f:
            f.write(str(dataset_path.absolute()))
        print(f"\n✅ 数据集路径已保存到: {config_file}")
        print("   (下一步会自动使用这个路径)")
        
    else:
        print(f"\n❌ 路径不存在: {dataset_path}")
        print("   请检查路径是否正确，或者重新下载数据集")
else:
    print("\n⏭️  跳过检查。请确保在运行下一步之前完成数据集下载。")

print("\n" + "=" * 60)
print("📝 下一步:")
print("   1. 确保数据集已经下载并解压")
print("   2. 记住数据集的路径")
print("   3. 运行: python step3_convert_dataset.py")
print("=" * 60)

