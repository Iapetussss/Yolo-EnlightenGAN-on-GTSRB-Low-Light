"""
生成纯低光照数据集（无任何增强）
用于 Baseline 实验
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import yaml

def create_lowlight_image(image, gamma_range=(0.3, 0.7)):
    """
    使用 Gamma 变换创建低光照图像
    
    Args:
        image: 输入图像
        gamma_range: Gamma 值范围
    
    Returns:
        lowlight_image: 低光照图像
        gamma_used: 使用的 gamma 值
    """
    # 随机选择 gamma 值
    gamma = np.random.uniform(*gamma_range)
    
    # 创建查找表
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 
        for i in range(256)
    ]).astype("uint8")
    
    # 应用 gamma 变换
    lowlight = cv2.LUT(image, table)
    
    return lowlight, gamma

def process_dataset(input_dir, output_dir, gamma_range=(0.3, 0.7)):
    """
    批量处理数据集
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 获取所有图像
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.ppm']:
        image_files.extend(list(input_path.rglob(f'*{ext}')))
    
    if not image_files:
        print(f"❌ 未找到图像文件")
        return False
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理每张图像
    success_count = 0
    gamma_values = []
    
    for img_file in tqdm(image_files, desc="生成低光照图像"):
        try:
            # 读取图像
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # 生成低光照版本
            lowlight, gamma_used = create_lowlight_image(image, gamma_range)
            gamma_values.append(gamma_used)
            
            # 保持相对路径结构
            rel_path = img_file.relative_to(input_path)
            out_file = output_path / rel_path
            
            # 创建输出目录
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为 PNG 格式并保存
            out_file_png = out_file.with_suffix('.png')
            cv2.imwrite(str(out_file_png), lowlight)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n处理失败 {img_file}: {e}")
    
    # 统计
    avg_gamma = np.mean(gamma_values)
    print(f"\n✅ 成功处理: {success_count}/{len(image_files)}")
    print(f"   平均 Gamma: {avg_gamma:.3f}")
    print(f"   Gamma 范围: [{min(gamma_values):.3f}, {max(gamma_values):.3f}]")
    
    return True

def copy_labels(source_label_dir, dest_label_dir):
    """
    复制标签文件
    """
    source_path = Path(source_label_dir)
    dest_path = Path(dest_label_dir)
    
    if not source_path.exists():
        print(f"⚠️  标签目录不存在: {source_label_dir}")
        return False
    
    # 获取所有标签文件
    label_files = list(source_path.rglob('*.txt'))
    
    if not label_files:
        print("⚠️  未找到标签文件")
        return False
    
    print(f"\n复制 {len(label_files)} 个标签文件...")
    
    for label_file in tqdm(label_files, desc="复制标签"):
        try:
            rel_path = label_file.relative_to(source_path)
            dest_file = dest_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_file, dest_file)
        except Exception as e:
            print(f"\n复制失败 {label_file}: {e}")
    
    print("✅ 标签复制完成")
    return True

def create_yolo_dataset(lowlight_base, labels_source, output_dir):
    """
    创建 YOLO 格式数据集
    """
    print("\n" + "=" * 70)
    print("创建 YOLO 格式数据集...")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建标准 YOLO 结构
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 移动图像和标签
    lowlight_path = Path(lowlight_base)
    labels_path = Path(labels_source)
    
    for split in ['train', 'val', 'test']:
        # 移动图像
        src_img_dir = lowlight_path / split
        dst_img_dir = output_path / 'images' / split
        
        if src_img_dir.exists():
            images = list(src_img_dir.glob('*.png'))
            print(f"\n{split.upper()}: 移动 {len(images)} 张图像...")
            
            for img in tqdm(images, desc=f"  图像"):
                dst = dst_img_dir / img.name
                shutil.copy2(img, dst)
        
        # 复制标签
        src_label_dir = labels_path / split
        dst_label_dir = output_path / 'labels' / split
        
        if src_label_dir.exists():
            labels = list(src_label_dir.glob('*.txt'))
            print(f"   复制 {len(labels)} 个标签...")
            
            for label in tqdm(labels, desc=f"  标签"):
                dst = dst_label_dir / label.name
                shutil.copy2(label, dst)
    
    print("\n✅ YOLO 数据集创建完成")
    print(f"   位置: {output_path}")
    
    return output_path

def create_config(dataset_path, config_path='configs/exp1_baseline.yaml'):
    """
    创建 YAML 配置文件
    """
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 43,
        'names': [
            "speed_20", "speed_30", "speed_50", "speed_60", "speed_70", "speed_80",
            "speed_80_end", "speed_100", "speed_120", "no_overtaking", "no_overtaking_trucks",
            "priority_at_next_intersection", "priority_road", "give_way", "stop", "no_entry",
            "no_entry_trucks", "no_entry_one_way", "general_caution", "dangerous_curve_left",
            "dangerous_curve_right", "double_curve", "bumpy_road", "slippery_road",
            "road_narrows_right", "road_works", "traffic_signals", "pedestrians",
            "children_crossing", "bicycles_crossing", "ice_or_snow", "wild_animals_crossing",
            "end_of_all_speed_and_overtaking_limits", "turn_right_ahead", "turn_left_ahead",
            "ahead_only", "go_straight_or_right", "go_straight_or_left", "keep_right",
            "keep_left", "roundabout_mandatory", "end_of_no_overtaking",
            "end_of_no_overtaking_trucks"
        ]
    }
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ 配置文件已创建: {config_file}")
    
    return config_file

def main():
    """主函数"""
    
    print("\n" + "=" * 70)
    print("  生成纯低光照数据集（Baseline 实验用）".center(70))
    print("=" * 70)
    
    print("\n说明:")
    print("  • 使用 Gamma 变换生成低光照图像")
    print("  • Gamma 范围: [0.3, 0.7]")
    print("  • 不进行任何增强处理")
    print("  • 用于建立性能基线")
    
    # 检查源数据
    print("\n" + "=" * 70)
    print("检查源数据...")
    print("=" * 70)
    
    # 寻找原始数据
    source_options = [
        ('yolo_dataset/images', 'yolo_dataset/labels'),
        ('traffic_sign_data/original/images', 'traffic_sign_data/labels'),
    ]
    
    source_img_dir = None
    source_label_dir = None
    
    for img_dir, label_dir in source_options:
        if Path(img_dir).exists() and Path(label_dir).exists():
            source_img_dir = img_dir
            source_label_dir = label_dir
            break
    
    if not source_img_dir:
        print("❌ 未找到源数据")
        print("\n提示: 将使用 yolo_dataset 作为源（虽然它已增强过）")
        print("     低光照化后仍可用于对比")
        source_img_dir = 'yolo_dataset/images'
        source_label_dir = 'yolo_dataset/labels'
    
    print(f"✅ 源图像: {source_img_dir}")
    print(f"✅ 源标签: {source_label_dir}")
    
    # 确认开始
    print("\n" + "=" * 70)
    print("⚠️  注意事项:")
    print("=" * 70)
    print("  • 处理时间: 约 30-60 分钟")
    print("  • 需要磁盘空间: 约 5GB")
    print("  • 输出目录: data/baseline_lowlight_dataset/")
    
    response = input("\n是否开始生成？(y/N): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    # 开始处理
    temp_lowlight = Path('data/temp_lowlight')
    
    # 步骤 1: 生成低光照图像
    print("\n" + "=" * 70)
    print("步骤 1/3: 生成低光照图像...")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        split_dir = Path(source_img_dir) / split
        if split_dir.exists():
            print(f"\n处理 {split.upper()} 集...")
            process_dataset(
                split_dir,
                temp_lowlight / split,
                gamma_range=(0.3, 0.7)
            )
    
    # 步骤 2: 创建 YOLO 数据集
    print("\n" + "=" * 70)
    print("步骤 2/3: 组织为 YOLO 格式...")
    print("=" * 70)
    
    final_dataset = create_yolo_dataset(
        temp_lowlight,
        source_label_dir,
        'data/baseline_lowlight_dataset'
    )
    
    # 步骤 3: 创建配置文件
    print("\n" + "=" * 70)
    print("步骤 3/3: 创建配置文件...")
    print("=" * 70)
    
    config_file = create_config(final_dataset)
    
    # 清理临时文件
    print("\n清理临时文件...")
    shutil.rmtree(temp_lowlight, ignore_errors=True)
    
    # 完成
    print("\n" + "=" * 70)
    print("  ✅ 纯低光照数据集生成完成！".center(70))
    print("=" * 70)
    
    print(f"\n数据集位置: {final_dataset}")
    print(f"配置文件: {config_file}")
    
    # 统计
    for split in ['train', 'val', 'test']:
        img_dir = final_dataset / 'images' / split
        if img_dir.exists():
            count = len(list(img_dir.glob('*.png')))
            print(f"  {split.upper():<6}: {count:>6} 张")
    
    print("\n🎯 下一步:")
    print("   运行 Baseline 实验:")
    print(f"   python scripts/training/train_baseline.py")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()

