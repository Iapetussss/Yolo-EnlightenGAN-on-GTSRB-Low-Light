"""
重组数据集以符合 YOLOv8 标准格式
"""

import shutil
from pathlib import Path

print("=" * 60)
print("🔧 重组数据集结构")
print("=" * 60)

# 源路径
data_root = Path('traffic_sign_data')
enhanced_images = data_root / 'enhanced_images'
labels_dir = data_root / 'labels'

# 目标路径
yolo_dataset = Path('yolo_dataset')

print(f"\n源目录: {data_root.absolute()}")
print(f"目标目录: {yolo_dataset.absolute()}")

# 创建目标目录结构
for split in ['train', 'val', 'test']:
    (yolo_dataset / 'images' / split).mkdir(parents=True, exist_ok=True)
    (yolo_dataset / 'labels' / split).mkdir(parents=True, exist_ok=True)

print("\n开始复制文件...")

# 复制文件
for split in ['train', 'val', 'test']:
    print(f"\n处理 {split} 集...")
    
    # 源路径
    src_images = enhanced_images / split
    src_labels = labels_dir / split
    
    # 目标路径
    dst_images = yolo_dataset / 'images' / split
    dst_labels = yolo_dataset / 'labels' / split
    
    if not src_images.exists():
        print(f"  ⚠️  图像目录不存在: {src_images}")
        continue
    
    # 复制图像
    image_files = list(src_images.glob('*.png'))
    print(f"  复制 {len(image_files)} 张图像...")
    for i, img_file in enumerate(image_files, 1):
        shutil.copy2(img_file, dst_images / img_file.name)
        if i % 1000 == 0:
            print(f"    已复制 {i}/{len(image_files)} 张图像...")
    
    # 复制标签
    if src_labels.exists():
        label_files = list(src_labels.glob('*.txt'))
        print(f"  复制 {len(label_files)} 个标签...")
        for i, label_file in enumerate(label_files, 1):
            shutil.copy2(label_file, dst_labels / label_file.name)
            if i % 1000 == 0:
                print(f"    已复制 {i}/{len(label_files)} 个标签...")
    else:
        print(f"  ⚠️  标签目录不存在: {src_labels}")
    
    print(f"  ✅ {split} 集完成: {len(image_files)} 图像")

print("\n" + "=" * 60)
print("✅ 数据集重组完成！")
print("=" * 60)

# 更新统计
for split in ['train', 'val', 'test']:
    img_count = len(list((yolo_dataset / 'images' / split).glob('*.png')))
    lbl_count = len(list((yolo_dataset / 'labels' / split).glob('*.txt')))
    print(f"{split.capitalize():5s}: {img_count} 图像, {lbl_count} 标签")

print(f"\n新数据集位置: {yolo_dataset.absolute()}")

