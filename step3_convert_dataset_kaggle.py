"""
步骤 3: 转换 Kaggle 版 GTSRB 数据集格式
将 Kaggle 版 GTSRB 格式转换为 YOLO 格式
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import shutil
import numpy as np

print("=" * 60)
print("🔄 步骤 3: 转换 Kaggle 版 GTSRB 数据集格式")
print("=" * 60)

# 读取保存的数据集路径
config_file = Path(__file__).parent / 'dataset_path.txt'

if config_file.exists():
    with open(config_file, 'r') as f:
        dataset_path = Path(f.read().strip())
    print(f"\n✅ 使用保存的数据集路径: {dataset_path}")
else:
    print("\n请输入 datasets 文件夹的路径:")
    print("例如: D:\\rgznzuoye\\new\\datasets")
    dataset_path = input("\n数据集路径: ").strip()
    
    if not dataset_path:
        print("❌ 必须提供路径")
        sys.exit(1)
    
    dataset_path = Path(dataset_path)

if not dataset_path.exists():
    print(f"\n❌ 路径不存在: {dataset_path}")
    sys.exit(1)

# 检查必要的目录
train_dir = dataset_path / 'Train'
test_dir = dataset_path / 'Test'
train_csv = dataset_path / 'Train.csv'
test_csv = dataset_path / 'Test.csv'

if not train_dir.exists():
    print(f"❌ 训练集目录不存在: {train_dir}")
    sys.exit(1)

if not test_dir.exists():
    print(f"❌ 测试集目录不存在: {test_dir}")
    sys.exit(1)

print(f"\n✅ 训练集: {train_dir}")
print(f"✅ 测试集: {test_dir}")

# 设置输出路径
output_root = dataset_path.parent / 'traffic_sign_data' / 'original'

print(f"\n输出路径: {output_root}")

# 确认
print("\n⚠️  注意:")
print("   - 转换过程可能需要 10-20 分钟")
print("   - 需要约 2-3 GB 的磁盘空间")
print("   - 会将图片复制并转换格式")
print("   - 会创建 YOLO 格式的标注文件")

response = input("\n是否继续? (输入 yes 继续): ").strip().lower()

if response != 'yes':
    print("\n❌ 用户取消操作")
    sys.exit(0)

print("\n" + "=" * 60)
print("开始转换数据集...")
print("=" * 60)

def convert_train_set(train_dir, train_csv, output_root):
    """转换训练集"""
    print("\n处理训练集...")
    
    # 创建输出目录
    train_images_out = output_root / 'images' / 'train'
    train_labels_out = output_root / 'labels' / 'train'
    train_images_out.mkdir(parents=True, exist_ok=True)
    train_labels_out.mkdir(parents=True, exist_ok=True)
    
    # 读取 CSV（如果存在）
    bbox_dict = {}
    if train_csv.exists():
        print("   读取 Train.csv 标注文件...")
        df = pd.read_csv(train_csv)
        
        # 构建路径到边界框的映射
        for _, row in df.iterrows():
            path = row['Path'] if 'Path' in df.columns else f"Train/{row['ClassId']}/{row['Path'].split('/')[-1]}"
            bbox_dict[path] = {
                'x1': row['Roi.X1'],
                'y1': row['Roi.Y1'],
                'x2': row['Roi.X2'],
                'y2': row['Roi.Y2'],
                'width': row['Width'],
                'height': row['Height'],
                'class_id': row['ClassId']
            }
    
    # 遍历所有类别文件夹
    image_count = 0
    
    for class_dir in tqdm(sorted(train_dir.iterdir()), desc="   转换训练集"):
        if not class_dir.is_dir():
            continue
        
        class_id = int(class_dir.name)
        
        # 遍历该类别的所有图片
        for img_file in class_dir.glob('*.png'):
            # 读取图片
            try:
                img = Image.open(img_file)
                width, height = img.size
                
                # 获取边界框信息
                relative_path = f"Train/{class_id}/{img_file.name}"
                
                if relative_path in bbox_dict:
                    bbox = bbox_dict[relative_path]
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    img_width, img_height = bbox['width'], bbox['height']
                else:
                    # 如果 CSV 中没有，使用整张图片作为边界框
                    x1, y1 = 0, 0
                    x2, y2 = width, height
                    img_width, img_height = width, height
                
                # 转换为 YOLO 格式（中心点坐标 + 归一化宽高）
                x_center = (x1 + x2) / 2.0 / img_width
                y_center = (y1 + y2) / 2.0 / img_height
                bbox_width = (x2 - x1) / img_width
                bbox_height = (y2 - y1) / img_height
                
                # 生成新文件名
                new_filename = f'train_{class_id:05d}_{image_count:06d}'
                
                # 保存图片
                dst_image = train_images_out / f'{new_filename}.png'
                img.save(dst_image)
                
                # 保存标注
                label_file = train_labels_out / f'{new_filename}.txt'
                with open(label_file, 'w') as f:
                    f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')
                
                image_count += 1
                
            except Exception as e:
                print(f"      警告: 处理图片 {img_file} 时出错: {e}")
                continue
    
    print(f"   ✅ 训练集转换完成: {image_count} 张图片")
    return image_count

def convert_test_set(test_dir, test_csv, output_root):
    """转换测试集"""
    print("\n处理测试集...")
    
    # 创建输出目录
    test_images_out = output_root / 'images' / 'test'
    test_labels_out = output_root / 'labels' / 'test'
    test_images_out.mkdir(parents=True, exist_ok=True)
    test_labels_out.mkdir(parents=True, exist_ok=True)
    
    # 读取 CSV
    if not test_csv.exists():
        print(f"   ⚠️  未找到 Test.csv，将使用图片本身作为边界框")
        # 遍历所有测试图片
        image_count = 0
        for img_file in tqdm(list(test_dir.glob('*.png')), desc="   转换测试集"):
            try:
                img = Image.open(img_file)
                width, height = img.size
                
                # 使用整张图片作为边界框，类别设为 0（需要后续手动标注）
                class_id = 0
                x_center, y_center = 0.5, 0.5
                bbox_width, bbox_height = 1.0, 1.0
                
                # 保存图片
                new_filename = f'test_{image_count:06d}'
                dst_image = test_images_out / f'{new_filename}.png'
                img.save(dst_image)
                
                # 保存标注
                label_file = test_labels_out / f'{new_filename}.txt'
                with open(label_file, 'w') as f:
                    f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')
                
                image_count += 1
            except Exception as e:
                print(f"      警告: 处理图片 {img_file} 时出错: {e}")
                continue
        
        print(f"   ✅ 测试集转换完成: {image_count} 张图片")
        return image_count
    
    # 如果有 CSV，使用 CSV 中的标注
    print("   读取 Test.csv 标注文件...")
    df = pd.read_csv(test_csv)
    
    image_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   转换测试集"):
        # 获取图片路径
        if 'Path' in df.columns:
            img_path = test_dir / Path(row['Path']).name
        else:
            # 假设图片在 Test 目录下
            img_path = test_dir / f"{row['Width']}_{row['Height']}.png"  # 需要根据实际情况调整
            # 或者尝试按索引查找
            possible_files = list(test_dir.glob('*.png'))
            if image_count < len(possible_files):
                img_path = possible_files[image_count]
        
        if not img_path.exists():
            # 尝试其他可能的文件名
            possible_names = [f"{image_count}.png", f"test_{image_count}.png", f"{image_count:05d}.png"]
            for name in possible_names:
                test_path = test_dir / name
                if test_path.exists():
                    img_path = test_path
                    break
        
        if not img_path.exists():
            print(f"      警告: 找不到图片 {img_path}")
            continue
        
        try:
            img = Image.open(img_path)
            
            # 读取边界框和类别
            class_id = int(row['ClassId']) if 'ClassId' in df.columns else 0
            width = int(row['Width'])
            height = int(row['Height'])
            x1 = int(row['Roi.X1'])
            y1 = int(row['Roi.Y1'])
            x2 = int(row['Roi.X2'])
            y2 = int(row['Roi.Y2'])
            
            # 转换为 YOLO 格式
            x_center = (x1 + x2) / 2.0 / width
            y_center = (y1 + y2) / 2.0 / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            
            # 保存图片
            new_filename = f'test_{image_count:06d}'
            dst_image = test_images_out / f'{new_filename}.png'
            img.save(dst_image)
            
            # 保存标注
            label_file = test_labels_out / f'{new_filename}.txt'
            with open(label_file, 'w') as f:
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')
            
            image_count += 1
            
        except Exception as e:
            print(f"      警告: 处理图片时出错: {e}")
            continue
    
    print(f"   ✅ 测试集转换完成: {image_count} 张图片")
    return image_count

def split_train_val(output_root, val_ratio=0.2):
    """从训练集中分割验证集"""
    print(f"\n分割验证集 (比例: {val_ratio})...")
    
    train_images = output_root / 'images' / 'train'
    train_labels = output_root / 'labels' / 'train'
    val_images = output_root / 'images' / 'val'
    val_labels = output_root / 'labels' / 'val'
    
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    # 获取所有训练图片
    all_images = list(train_images.glob('*.png'))
    
    # 随机打乱
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    # 分割
    val_size = int(len(all_images) * val_ratio)
    val_images_list = all_images[:val_size]
    
    # 移动文件
    for img_path in tqdm(val_images_list, desc="   移动验证集"):
        # 移动图片
        dst_img = val_images / img_path.name
        shutil.move(str(img_path), str(dst_img))
        
        # 移动标注
        label_path = train_labels / f'{img_path.stem}.txt'
        if label_path.exists():
            dst_label = val_labels / label_path.name
            shutil.move(str(label_path), str(dst_label))
    
    train_count = len(list(train_images.glob('*.png')))
    val_count = len(list(val_images.glob('*.png')))
    
    print(f"   ✅ 训练集: {train_count} 张")
    print(f"   ✅ 验证集: {val_count} 张")

try:
    # 转换训练集
    train_count = convert_train_set(train_dir, train_csv, output_root)
    
    # 转换测试集
    test_count = convert_test_set(test_dir, test_csv, output_root)
    
    # 分割验证集
    split_train_val(output_root, val_ratio=0.2)
    
    print("\n" + "=" * 60)
    print("✅ 数据集转换完成！")
    print("=" * 60)
    
    # 统计信息
    train_images = len(list((output_root / 'images' / 'train').glob('*.png')))
    val_images = len(list((output_root / 'images' / 'val').glob('*.png')))
    test_images = len(list((output_root / 'images' / 'test').glob('*.png')))
    
    print(f"\n数据集统计:")
    print(f"  训练集: {train_images} 张图片")
    print(f"  验证集: {val_images} 张图片")
    print(f"  测试集: {test_images} 张图片")
    print(f"  总计: {train_images + val_images + test_images} 张图片")
    
    # 保存输出路径
    output_config = Path(__file__).parent / 'converted_dataset_path.txt'
    with open(output_config, 'w') as f:
        f.write(str(output_root.absolute()))
    
    print(f"\n输出路径已保存: {output_config}")
    print(f"数据集位置: {output_root}")
    
    print("\n" + "=" * 60)
    print("📝 下一步:")
    print("=" * 60)
    print("运行以下命令创建低光照数据集:")
    print("   python step4_create_lowlight.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 转换过程中出现错误:")
    print("=" * 60)
    print(str(e))
    import traceback
    traceback.print_exc()
    print("\n请检查错误信息并修正后重试")
    sys.exit(1)

