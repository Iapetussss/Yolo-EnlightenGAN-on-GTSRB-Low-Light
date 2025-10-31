"""
GTSRB 数据集准备和转换脚本
将 GTSRB 数据集转换为 YOLO 格式
"""

import os
import cv2
import csv
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image


class GTSRBDatasetConverter:
    """
    GTSRB 数据集转换器
    将 GTSRB 格式转换为 YOLO 格式
    """
    
    def __init__(self, gtsrb_root, output_root):
        """
        初始化转换器
        
        Args:
            gtsrb_root: GTSRB 数据集根目录
            output_root: 输出目录
        """
        self.gtsrb_root = Path(gtsrb_root)
        self.output_root = Path(output_root)
        self.num_classes = 43
        
    def convert_train_set(self):
        """
        转换训练集
        GTSRB 训练集结构: 
        - GTSRB/Final_Training/Images/00000/*.ppm
        - 每个类别一个文件夹
        """
        print("正在转换训练集...")
        
        train_images = self.output_root / 'images' / 'train'
        train_labels = self.output_root / 'labels' / 'train'
        train_images.mkdir(parents=True, exist_ok=True)
        train_labels.mkdir(parents=True, exist_ok=True)
        
        # 遍历所有类别文件夹
        train_path = self.gtsrb_root / 'Final_Training' / 'Images'
        
        if not train_path.exists():
            print(f"警告: 训练集路径不存在: {train_path}")
            return
            
        class_folders = sorted([f for f in train_path.iterdir() if f.is_dir()])
        
        image_count = 0
        for class_folder in tqdm(class_folders, desc="处理类别"):
            class_id = int(class_folder.name)
            
            # 读取 CSV 标注文件
            csv_file = class_folder / f'GT-{class_folder.name}.csv'
            
            if not csv_file.exists():
                print(f"警告: CSV 文件不存在: {csv_file}")
                continue
                
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=';')
                
                for row in reader:
                    # 获取图像信息
                    filename = row['Filename']
                    width = int(row['Width'])
                    height = int(row['Height'])
                    x1 = int(row['Roi.X1'])
                    y1 = int(row['Roi.Y1'])
                    x2 = int(row['Roi.X2'])
                    y2 = int(row['Roi.Y2'])
                    
                    # 转换为 YOLO 格式 (中心点坐标 + 宽高，归一化)
                    x_center = (x1 + x2) / 2.0 / width
                    y_center = (y1 + y2) / 2.0 / height
                    bbox_width = (x2 - x1) / width
                    bbox_height = (y2 - y1) / height
                    
                    # 复制图像
                    src_image = class_folder / filename
                    if not src_image.exists():
                        continue
                        
                    # 生成新的文件名
                    new_filename = f'train_{class_id:05d}_{image_count:06d}'
                    dst_image = train_images / f'{new_filename}.png'
                    
                    # 转换图像格式为 PNG (GTSRB 原始为 PPM)
                    img = Image.open(src_image)
                    img.save(dst_image)
                    
                    # 保存标注
                    label_file = train_labels / f'{new_filename}.txt'
                    with open(label_file, 'w') as lf:
                        lf.write(f'{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')
                    
                    image_count += 1
        
        print(f"训练集转换完成！共 {image_count} 张图像")
    
    def convert_test_set(self):
        """
        转换测试集
        GTSRB 测试集结构:
        - GTSRB/Final_Test/Images/*.ppm
        - GT-final_test.csv (包含所有标注)
        """
        print("正在转换测试集...")
        
        test_images = self.output_root / 'images' / 'test'
        test_labels = self.output_root / 'labels' / 'test'
        test_images.mkdir(parents=True, exist_ok=True)
        test_labels.mkdir(parents=True, exist_ok=True)
        
        # 读取测试集 CSV
        test_path = self.gtsrb_root / 'Final_Test' / 'Images'
        csv_file = test_path / 'GT-final_test.csv'
        
        if not csv_file.exists():
            print(f"警告: 测试集 CSV 不存在: {csv_file}")
            return
        
        image_count = 0
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            
            for row in tqdm(list(reader), desc="处理测试图像"):
                # 获取图像信息
                filename = row['Filename']
                width = int(row['Width'])
                height = int(row['Height'])
                x1 = int(row['Roi.X1'])
                y1 = int(row['Roi.Y1'])
                x2 = int(row['Roi.X2'])
                y2 = int(row['Roi.Y2'])
                class_id = int(row['ClassId'])
                
                # 转换为 YOLO 格式
                x_center = (x1 + x2) / 2.0 / width
                y_center = (y1 + y2) / 2.0 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                # 复制图像
                src_image = test_path / filename
                if not src_image.exists():
                    continue
                
                # 生成新的文件名
                new_filename = f'test_{image_count:06d}'
                dst_image = test_images / f'{new_filename}.png'
                
                # 转换图像格式
                img = Image.open(src_image)
                img.save(dst_image)
                
                # 保存标注
                label_file = test_labels / f'{new_filename}.txt'
                with open(label_file, 'w') as lf:
                    lf.write(f'{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')
                
                image_count += 1
        
        print(f"测试集转换完成！共 {image_count} 张图像")
    
    def split_train_val(self, val_ratio=0.2):
        """
        从训练集中分割出验证集
        
        Args:
            val_ratio: 验证集比例
        """
        print(f"正在分割验证集 (比例: {val_ratio})...")
        
        train_images = self.output_root / 'images' / 'train'
        train_labels = self.output_root / 'labels' / 'train'
        val_images = self.output_root / 'images' / 'val'
        val_labels = self.output_root / 'labels' / 'val'
        
        val_images.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)
        
        # 获取所有训练图像
        all_images = list(train_images.glob('*.png'))
        
        # 随机打乱
        np.random.seed(42)
        np.random.shuffle(all_images)
        
        # 分割
        val_size = int(len(all_images) * val_ratio)
        val_images_list = all_images[:val_size]
        
        # 移动文件
        for img_path in tqdm(val_images_list, desc="移动验证集"):
            # 移动图像
            dst_img = val_images / img_path.name
            shutil.move(str(img_path), str(dst_img))
            
            # 移动标注
            label_path = train_labels / f'{img_path.stem}.txt'
            if label_path.exists():
                dst_label = val_labels / label_path.name
                shutil.move(str(label_path), str(dst_label))
        
        print(f"验证集分割完成！训练集: {len(list(train_images.glob('*.png')))} 张，验证集: {len(list(val_images.glob('*.png')))} 张")
    
    def convert_all(self, val_ratio=0.2):
        """
        执行完整的数据集转换流程
        
        Args:
            val_ratio: 验证集比例
        """
        print("开始转换 GTSRB 数据集到 YOLO 格式...")
        print(f"输入: {self.gtsrb_root}")
        print(f"输出: {self.output_root}")
        
        # 转换训练集
        self.convert_train_set()
        
        # 转换测试集
        self.convert_test_set()
        
        # 分割验证集
        self.split_train_val(val_ratio)
        
        print("\n数据集转换完成！")
        self.print_statistics()
    
    def print_statistics(self):
        """
        打印数据集统计信息
        """
        print("\n=== 数据集统计 ===")
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_root / 'images' / split
            labels_dir = self.output_root / 'labels' / split
            
            if images_dir.exists():
                num_images = len(list(images_dir.glob('*.png')))
                num_labels = len(list(labels_dir.glob('*.txt')))
                print(f"{split.capitalize():5s}: {num_images} 张图像, {num_labels} 个标注文件")


def create_low_light_images(input_dir, output_dir, gamma_values=[0.3, 0.5, 0.7]):
    """
    创建低光照图像
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出图像目录
        gamma_values: Gamma 值列表 (越小越暗)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    print(f"正在创建低光照图像: {len(image_files)} 张")
    
    for img_file in tqdm(image_files, desc="处理图像"):
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # 随机选择一个 gamma 值
        gamma = np.random.choice(gamma_values)
        
        # 应用 gamma 变换
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        low_light = cv2.LUT(image, table)
        
        # 保存
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), low_light)
    
    print("低光照图像创建完成！")


# 使用示例
if __name__ == "__main__":
    # 设置路径
    GTSRB_ROOT = "path/to/GTSRB"  # 修改为你的 GTSRB 数据集路径
    OUTPUT_ROOT = "../traffic_sign_data/original"
    
    # 步骤 1: 转换数据集格式
    print("\n=== 步骤 1: 转换 GTSRB 数据集到 YOLO 格式 ===")
    converter = GTSRBDatasetConverter(GTSRB_ROOT, OUTPUT_ROOT)
    converter.convert_all(val_ratio=0.2)
    
    # 步骤 2: 创建低光照版本
    print("\n=== 步骤 2: 创建低光照数据集 ===")
    for split in ['train', 'val', 'test']:
        input_dir = f"{OUTPUT_ROOT}/images/{split}"
        output_dir = f"../traffic_sign_data/low_light/images/{split}"
        
        if os.path.exists(input_dir):
            create_low_light_images(input_dir, output_dir)
            
            # 复制标注文件
            src_labels = Path(f"{OUTPUT_ROOT}/labels/{split}")
            dst_labels = Path(f"../traffic_sign_data/low_light/labels/{split}")
            dst_labels.mkdir(parents=True, exist_ok=True)
            
            for label_file in src_labels.glob('*.txt'):
                shutil.copy(str(label_file), str(dst_labels / label_file.name))
    
    print("\n数据准备完成！")

