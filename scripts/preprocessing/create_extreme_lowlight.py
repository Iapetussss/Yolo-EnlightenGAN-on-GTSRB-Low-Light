#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成更极端的低光照数据集
用于真正的低光照Baseline实验
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import yaml

def create_extreme_lowlight(image, gamma=0.25):
    """
    使用更极端的 Gamma 变换创建低光照图像
    
    Args:
        image: 输入图像
        gamma: Gamma 值（越小越暗，0.2-0.3为极暗）
    
    Returns:
        lowlight_image: 低光照图像
    """
    # 创建查找表（gamma越小，图像越暗）
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 
        for i in range(256)
    ]).astype("uint8")
    
    # 应用 gamma 变换
    lowlight = cv2.LUT(image, table)
    
    return lowlight

def process_dataset(input_dir, output_dir, gamma=0.25):
    """
    批量处理数据集，生成极暗的低光照图像
    
    Args:
        input_dir: 原始图像目录（YOLO格式：images/train, images/val, images/test）
        output_dir: 输出目录
        gamma: Gamma值（0.2=极暗, 0.25=很暗, 0.3=偏暗）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 检查输入目录
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    print("="*60)
    print("🌑 生成极端低光照数据集")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Gamma值: {gamma} (越小越暗)")
    print(f"\n评估标准:")
    print(f"  gamma=0.2: 极暗 (几乎看不见)")
    print(f"  gamma=0.25: 很暗 (有挑战性) ⭐推荐")
    print(f"  gamma=0.3: 偏暗 (相对容易)")
    print(f"  gamma=0.4+: 微暗 (基本正常)")
    print()
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        img_dir = output_path / 'images' / split
        label_dir = output_path / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个split
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        input_img_dir = input_path / 'images' / split
        input_label_dir = input_path / 'labels' / split
        
        if not input_img_dir.exists():
            print(f"⚠️ 跳过 {split}（目录不存在）")
            continue
        
        output_img_dir = output_path / 'images' / split
        output_label_dir = output_path / 'labels' / split
        
        # 获取所有图像
        images = sorted(input_img_dir.glob("*.png"))
        if len(images) == 0:
            images = sorted(input_img_dir.glob("*.jpg"))
        
        print(f"\n处理 {split} 集...")
        print(f"  找到 {len(images)} 张图像")
        print(f"  开始处理...")
        
        processed = 0
        for i, img_path in enumerate(images, 1):
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # 生成低光照图像
            lowlight_img = create_extreme_lowlight(img, gamma)
            
            # 保存图像
            output_img_path = output_img_dir / img_path.name
            cv2.imwrite(str(output_img_path), lowlight_img)
            
            # 复制标签文件
            label_path = input_label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                output_label_path = output_label_dir / label_path.name
                shutil.copy2(label_path, output_label_path)
            
            processed += 1
            
            # 每500张显示一次进度（更频繁）
            if processed % 500 == 0 or processed == len(images):
                percent = processed / len(images) * 100
                print(f"  [{percent:5.1f}%] 已处理: {processed}/{len(images)}")
        
        print(f"  ✅ {split} 完成: {processed} 张图像")
        total_images += processed
    
    print("\n" + "="*60)
    print(f"✅ 总共处理了 {total_images} 张图像")
    print(f"📂 输出目录: {output_path}")
    
    # 创建YAML配置文件
    config_path = output_path / 'dataset.yaml'
    config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 43,
        'names': list(range(43))
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📄 配置文件已创建: {config_path}")
    
    # 测试一张图像
    print("\n" + "="*60)
    print("🧪 测试生成的低光照图像...")
    test_img_path = output_path / 'images' / 'train' / next(iter(output_path.glob('images/train/*.png')), None)
    if test_img_path and test_img_path.exists():
        test_img = cv2.imread(str(test_img_path))
        brightness = np.mean(test_img)
        print(f"测试图像平均亮度: {brightness:.1f} / 255 ({brightness/255*100:.1f}%)")
        
        if brightness < 50:
            print("✅ 极暗 - 真正的低光照！")
        elif brightness < 100:
            print("✅ 很暗 - 有挑战性的低光照")
        elif brightness < 150:
            print("⚠️ 偏暗 - 仍然偏亮")
        else:
            print("❌ 太亮 - 需要降低gamma值")
    
    return True

def main():
    """主函数"""
    print("="*60)
    print("🌑 生成极端低光照数据集")
    print("="*60)
    print("\n请选择 Gamma 值:")
    print("  [1] gamma=0.2  (极暗，几乎看不见)")
    print("  [2] gamma=0.25 (很暗，有挑战性) ⭐推荐")
    print("  [3] gamma=0.3  (偏暗，相对容易)")
    print("  [4] 自定义")
    
    choice = input("\n请选择 [1/2/3/4]: ").strip()
    
    if choice == '1':
        gamma = 0.2
    elif choice == '2':
        gamma = 0.25
    elif choice == '3':
        gamma = 0.3
    elif choice == '4':
        gamma_input = input("请输入gamma值 (0.1-0.4): ").strip()
        try:
            gamma = float(gamma_input)
            if gamma < 0.1 or gamma > 0.4:
                print("⚠️ gamma值超出范围，使用0.25")
                gamma = 0.25
        except:
            print("⚠️ 输入无效，使用0.25")
            gamma = 0.25
    else:
        print("⚠️ 无效选择，使用默认值0.25")
        gamma = 0.25
    
    # 确定输入目录
    print("\n" + "="*60)
    print("选择输入数据源:")
    print("  [1] yolo_dataset (原始标准图像)")
    print("  [2] traffic_sign_data/enhanced_images (增强后的图像)")
    print("  [3] 自定义路径")
    
    src_choice = input("\n请选择 [1/2/3]: ").strip()
    
    if src_choice == '1':
        input_dir = Path("data/yolo_dataset")
    elif src_choice == '2':
        input_dir = Path("traffic_sign_data/enhanced_images")
    elif src_choice == '3':
        input_dir = Path(input("请输入路径: ").strip())
    else:
        print("⚠️ 无效选择，使用 yolo_dataset")
        input_dir = Path("data/yolo_dataset")
    
    # 输出目录
    output_dir = Path("data/baseline_lowlight_dataset")
    
    print("\n" + "="*60)
    print("⚠️ 警告")
    print("="*60)
    print(f"输出目录: {output_dir}")
    if output_dir.exists():
        print("⚠️ 目录已存在，将会覆盖！")
        confirm = input("确认继续？(y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 已取消")
            return
        
        # 删除旧目录
        shutil.rmtree(output_dir)
        print("✅ 已清理旧数据")
    
    # 开始处理
    print("\n开始处理...")
    success = process_dataset(input_dir, output_dir, gamma)
    
    if success:
        print("\n" + "="*60)
        print("✅ 完成！")
        print("="*60)
        print("\n下一步:")
        print("  1. 停止当前训练（如果有）")
        print("  2. 运行: python scripts/training/train_baseline.py")
        print("  3. 使用新的极端低光照数据训练")
        print("\n预期效果:")
        print("  • Baseline mAP可能降到 70-85%")
        print("  • 增强方法的优势会更明显")
        print("  • 实验更有说服力！")
    else:
        print("\n❌ 处理失败，请检查错误信息")

if __name__ == '__main__':
    main()

