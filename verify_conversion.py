import os
import csv

# 设置路径
DEST_DIR = "D:/datasets/GTSRB"

def verify_directory_structure():
    # 检查主要目录是否存在
    directories_to_check = [
        os.path.join(DEST_DIR, "Final_Training", "Images"),
        os.path.join(DEST_DIR, "Final_Test", "Images")
    ]
    
    for dir_path in directories_to_check:
        if os.path.exists(dir_path):
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"✗ 目录不存在: {dir_path}")
    
    # 检查一些随机选择的类别目录
    sample_classes = ["00000", "00010", "00020", "00030", "00040"]
    for class_id in sample_classes:
        class_dir = os.path.join(DEST_DIR, "Final_Training", "Images", class_id)
        if os.path.exists(class_dir):
            files = os.listdir(class_dir)
            ppm_files = [f for f in files if f.endswith('.ppm')]
            gt_file = f"GT-{class_id}.csv"
            
            print(f"✓ 类别目录 {class_id} 存在")
            print(f"  - PPM文件数量: {len(ppm_files)}")
            print(f"  - GT文件存在: {gt_file in files}")
        else:
            print(f"✗ 类别目录 {class_id} 不存在")
    
    # 检查测试集GT文件
    test_gt_file = os.path.join(DEST_DIR, "Final_Test", "Images", "GT-final_test.csv")
    if os.path.exists(test_gt_file):
        print(f"✓ 测试集GT文件存在: {test_gt_file}")
    else:
        print(f"✗ 测试集GT文件不存在: {test_gt_file}")

def check_csv_format():
    # 检查一个训练集GT文件格式
    sample_class = "00000"
    gt_file_path = os.path.join(DEST_DIR, "Final_Training", "Images", sample_class, f"GT-{sample_class}.csv")
    
    if os.path.exists(gt_file_path):
        print(f"\n检查 {gt_file_path} 文件格式:")
        try:
            with open(gt_file_path, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                header = next(reader)
                print(f"  - 表头: {header}")
                # 检查几行数据
                for i, row in enumerate(reader):
                    if i < 3:
                        print(f"  - 数据行 {i+1}: {row}")
                    else:
                        break
        except Exception as e:
            print(f"  - 读取CSV文件时出错: {e}")

def main():
    print("开始验证数据集转换...")
    verify_directory_structure()
    check_csv_format()
    print("\n验证完成！请根据上面的输出确认数据集结构是否正确。")
    print("注意：如果路径中出现权限错误，可能是因为脚本无法访问D盘，但这并不意味着转换失败。")

if __name__ == "__main__":
    main()