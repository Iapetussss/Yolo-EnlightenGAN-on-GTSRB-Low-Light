import os
import csv
from PIL import Image
import shutil

# 设置路径
SOURCE_DIR = "z:/CodeLibraryI/HybridProject/School_AI_class_report/archive"
DEST_DIR = "D:/datasets/GTSRB"

# 创建目标目录结构
def create_directory_structure():
    # 创建训练集目录
    os.makedirs(os.path.join(DEST_DIR, "Final_Training", "Images"), exist_ok=True)
    # 创建测试集目录
    os.makedirs(os.path.join(DEST_DIR, "Final_Test", "Images"), exist_ok=True)
    print("目录结构创建完成")

# 处理训练集
def process_training_data():
    train_csv_path = os.path.join(SOURCE_DIR, "Train.csv")
    train_data = {}
    
    # 读取训练集CSV文件
    with open(train_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = row["ClassId"]
            if class_id not in train_data:
                train_data[class_id] = []
            train_data[class_id].append(row)
    
    # 处理每个类别
    for class_id, images in train_data.items():
        # 格式化为5位数字
        class_dir_name = f"{int(class_id):05d}"
        class_dir = os.path.join(DEST_DIR, "Final_Training", "Images", class_dir_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 准备创建GT文件
        gt_file_path = os.path.join(class_dir, f"GT-{class_dir_name}.csv")
        
        with open(gt_file_path, 'w', newline='') as gt_file:
            gt_writer = csv.writer(gt_file, delimiter=';')
            # 写入GT文件头
            gt_writer.writerow(["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"])
            
            # 处理该类别的所有图像
            for i, image_data in enumerate(images):
                # 原始图像路径
                src_path = os.path.join(SOURCE_DIR, image_data["Path"])
                # 目标文件名格式：00000_00000.ppm（类别_序号.ppm）
                dst_filename = f"{class_dir_name}_{i:05d}.ppm"
                dst_path = os.path.join(class_dir, dst_filename)
                
                # 转换图像格式（PNG -> PPM）
                try:
                    img = Image.open(src_path)
                    img.save(dst_path, format="PPM")
                    
                    # 写入GT文件行
                    gt_writer.writerow([
                        dst_filename,
                        image_data["Width"],
                        image_data["Height"],
                        image_data["Roi.X1"],
                        image_data["Roi.Y1"],
                        image_data["Roi.X2"],
                        image_data["Roi.Y2"],
                        image_data["ClassId"]
                    ])
                except Exception as e:
                    print(f"处理文件 {src_path} 时出错: {e}")
        
        print(f"类别 {class_id} 处理完成，共 {len(images)} 张图像")

# 处理测试集
def process_testing_data():
    test_csv_path = os.path.join(SOURCE_DIR, "Test.csv")
    test_dir = os.path.join(DEST_DIR, "Final_Test", "Images")
    
    # 准备创建GT文件
    gt_file_path = os.path.join(test_dir, "GT-final_test.csv")
    
    with open(gt_file_path, 'w', newline='') as gt_file:
        gt_writer = csv.writer(gt_file, delimiter=';')
        # 写入GT文件头
        gt_writer.writerow(["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"])
        
        # 读取测试集CSV文件
        with open(test_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 原始图像路径
                src_path = os.path.join(SOURCE_DIR, row["Path"])
                # 目标文件名保持不变，但更改扩展名
                filename = os.path.basename(row["Path"])
                filename_without_ext = os.path.splitext(filename)[0]
                dst_filename = f"{filename_without_ext}.ppm"
                dst_path = os.path.join(test_dir, dst_filename)
                
                # 转换图像格式（PNG -> PPM）
                try:
                    img = Image.open(src_path)
                    img.save(dst_path, format="PPM")
                    
                    # 写入GT文件行
                    gt_writer.writerow([
                        dst_filename,
                        row["Width"],
                        row["Height"],
                        row["Roi.X1"],
                        row["Roi.Y1"],
                        row["Roi.X2"],
                        row["Roi.Y2"],
                        row["ClassId"]
                    ])
                except Exception as e:
                    print(f"处理文件 {src_path} 时出错: {e}")
    
    print("测试集处理完成")

# 主函数
def main():
    print("开始转换数据集...")
    create_directory_structure()
    
    print("处理训练集...")
    process_training_data()
    
    print("处理测试集...")
    process_testing_data()
    
    print("数据集转换完成！")
    print(f"转换后的数据集保存在: {DEST_DIR}")

if __name__ == "__main__":
    main()