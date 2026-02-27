"""
步骤 4: 创建低光照数据集
通过 Gamma 变换模拟低光照环境
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

print("=" * 60)
print("🌙 步骤 4: 创建低光照数据集")
print("=" * 60)

# 读取上一步的输出路径
config_file = Path(__file__).parent / 'converted_dataset_path.txt'

if config_file.exists():
    with open(config_file, 'r') as f:
        input_root = Path(f.read().strip())
    print(f"\n✅ 使用上一步转换的数据集: {input_root}")
else:
    print("\n⚠️  未找到上一步的输出路径配置")
    print("请输入转换后的数据集路径:")
    print("例如: D:\\rgznzuoye\\traffic_sign_data\\original")
    input_path = input("\n数据集路径: ").strip()
    
    if not input_path:
        print("❌ 必须提供路径")
        sys.exit(1)
    
    input_root = Path(input_path)

if not input_root.exists():
    print(f"\n❌ 路径不存在: {input_root}")
    sys.exit(1)

# 设置输出路径
output_root = input_root.parent / 'low_light'

print(f"\n输入路径: {input_root}")
print(f"输出路径: {output_root}")

# 说明低光照是如何创建的
print("\n" + "=" * 60)
print("📚 低光照图像生成原理:")
print("=" * 60)
print("""
我们使用 Gamma 变换来模拟低光照环境:

- Gamma < 1.0: 图像变暗
- Gamma = 1.0: 图像不变
- Gamma > 1.0: 图像变亮

在这个项目中，我们随机选择 Gamma 值在 0.3-0.7 之间，
这样可以模拟不同程度的低光照环境。

例如:
- Gamma = 0.3: 非常暗（夜晚）
- Gamma = 0.5: 较暗（阴天或傍晚）
- Gamma = 0.7: 稍暗（室内光线不足）
""")

# 确认
print("\n⚠️  注意:")
print("   - 创建过程可能需要 10-20 分钟")
print("   - 需要约 2-3 GB 的额外磁盘空间")
print("   - 会为每个数据集划分（train/val/test）创建低光照版本")

response = input("\n是否继续? (输入 yes 继续): ").strip().lower()

if response != 'yes':
    print("\n❌ 用户取消操作")
    sys.exit(0)

# 创建低光照图像的函数
def create_low_light_images(input_dir, output_dir, gamma_range=(0.3, 0.7)):
    """创建低光照图像"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    if not image_files:
        print(f"   ⚠️  未找到图像文件: {input_path}")
        return 0
    
    print(f"   找到 {len(image_files)} 张图像")
    
    for img_file in tqdm(image_files, desc=f"   处理 {input_path.name}"):
        # 读取图像
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # 随机选择 gamma 值
        gamma = np.random.uniform(*gamma_range)
        
        # 应用 gamma 变换
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        low_light = cv2.LUT(image, table)
        
        # 保存
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), low_light)
    
    return len(image_files)

# 开始创建
print("\n" + "=" * 60)
print("开始创建低光照数据集...")
print("=" * 60)

total_images = 0

try:
    for split in ['train', 'val', 'test']:
        print(f"\n处理 {split} 集...")
        
        # 创建低光照图像
        input_images = input_root / 'images' / split
        output_images = output_root / 'images' / split
        
        if input_images.exists():
            count = create_low_light_images(input_images, output_images)
            total_images += count
            print(f"   ✅ 完成 {count} 张图像")
            
            # 复制标注文件
            src_labels = input_root / 'labels' / split
            dst_labels = output_root / 'labels' / split
            dst_labels.mkdir(parents=True, exist_ok=True)
            
            if src_labels.exists():
                label_files = list(src_labels.glob('*.txt'))
                for label_file in label_files:
                    shutil.copy(str(label_file), str(dst_labels / label_file.name))
                print(f"   ✅ 复制 {len(label_files)} 个标注文件")
        else:
            print(f"   ⚠️  跳过 {split} (目录不存在)")
    
    print("\n" + "=" * 60)
    print("✅ 低光照数据集创建完成！")
    print("=" * 60)
    print(f"\n总共处理: {total_images} 张图像")
    print(f"输出位置: {output_root}")
    
    # 保存输出路径
    output_config = Path(__file__).parent / 'lowlight_dataset_path.txt'
    with open(output_config, 'w') as f:
        f.write(str(output_root.absolute()))
    
    print("\n下一步: 运行 python step5_enhance_images.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 创建过程中出现错误:")
    print("=" * 60)
    print(str(e))
    print("\n请检查错误信息并修正后重试")
    sys.exit(1)

