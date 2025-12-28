"""
步骤 5: 增强低光照图像
使用传统方法 (CLAHE + Gamma 校正) 增强图像
"""

import sys
from pathlib import Path
import shutil

print("=" * 60)
print("✨ 步骤 5: 增强低光照图像")
print("=" * 60)

# 读取上一步的输出路径
config_file = Path(__file__).parent / 'lowlight_dataset_path.txt'

if config_file.exists():
    with open(config_file, 'r') as f:
        input_root = Path(f.read().strip())
    print(f"\n✅ 使用上一步的低光照数据集: {input_root}")
else:
    print("\n⚠️  未找到上一步的输出路径配置")
    print("请输入低光照数据集路径:")
    print("例如: D:\\rgznzuoye\\traffic_sign_data\\low_light")
    input_path = input("\n数据集路径: ").strip()
    
    if not input_path:
        print("❌ 必须提供路径")
        sys.exit(1)
    
    input_root = Path(input_path)

if not input_root.exists():
    print(f"\n❌ 路径不存在: {input_root}")
    sys.exit(1)

# 设置输出路径
output_root = input_root.parent / 'enhanced_images'

print(f"\n输入路径: {input_root}")
print(f"输出路径: {output_root}")

# 说明增强方法
print("\n" + "=" * 60)
print("📚 图像增强方法:")
print("=" * 60)
print("""
我们使用传统的图像增强方法 (不需要下载额外模型):

1. CLAHE (对比度限制自适应直方图均衡)
   - 增强局部对比度
   - 避免过度增强

2. Gamma 校正
   - 调整整体亮度
   - 使暗的区域更明亮

这种方法虽然不如 EnlightenGAN 效果好，但:
✅ 不需要下载大模型
✅ 运行速度快
✅ 效果也不错
✅ 适合初学者

如果将来想尝试 EnlightenGAN，可以修改代码中的参数。
""")

# 确认
print("\n⚠️  注意:")
print("   - 增强过程可能需要 20-40 分钟")
print("   - 需要约 2-3 GB 的额外磁盘空间")
print("   - 会为所有图像生成增强版本")

response = input("\n是否继续? (输入 yes 继续): ").strip().lower()

if response != 'yes':
    print("\n❌ 用户取消操作")
    sys.exit(0)

# 开始增强
print("\n" + "=" * 60)
print("开始增强图像...")
print("=" * 60)

try:
    from enlightened_gtsrb import GTSRBEnlightenGANDetector
    
    # 创建检测器
    detector = GTSRBEnlightenGANDetector()
    
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        input_images = input_root / 'images' / split
        output_images = output_root / split
        
        if not input_images.exists():
            print(f"\n⚠️  跳过 {split} (目录不存在)")
            continue
        
        print(f"\n增强 {split} 集...")
        
        # 增强图像
        detector.enhance_dataset(
            input_dir=str(input_images),
            output_dir=str(output_images),
            method='traditional'  # 使用传统方法
        )
        
        # 复制标注文件
        src_labels = input_root / 'labels' / split
        dst_labels = output_root.parent / 'labels' / split
        dst_labels.mkdir(parents=True, exist_ok=True)
        
        if src_labels.exists():
            label_files = list(src_labels.glob('*.txt'))
            for label_file in label_files:
                shutil.copy(str(label_file), str(dst_labels / label_file.name))
            print(f"✅ 复制 {len(label_files)} 个标注文件")
        
        # 统计
        enhanced_count = len(list(output_images.glob('*.png')))
        total_images += enhanced_count
        print(f"✅ {split} 集完成: {enhanced_count} 张图像")
    
    print("\n" + "=" * 60)
    print("✅ 图像增强完成！")
    print("=" * 60)
    print(f"\n总共增强: {total_images} 张图像")
    print(f"输出位置: {output_root}")
    
    # 保存输出路径
    output_config = Path(__file__).parent / 'enhanced_dataset_path.txt'
    with open(output_config, 'w') as f:
        f.write(str(output_root.absolute()))
    
    # 更新 YAML 配置文件
    print("\n" + "=" * 60)
    print("更新 YOLOv8 配置文件...")
    print("=" * 60)
    
    yaml_path = Path(__file__).parent.parent / 'traffic_signs.yaml'
    
    # 计算相对路径
    try:
        rel_train = output_root / 'train'
        rel_val = output_root / 'val'
        rel_test = output_root / 'test'
        
        # 读取现有配置
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 备份原文件
            backup_path = yaml_path.parent / 'traffic_signs.yaml.backup'
            shutil.copy(yaml_path, backup_path)
            print(f"✅ 原配置已备份到: {backup_path}")
            
            # 更新路径（保持原有格式）
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith('train:'):
                    new_lines.append(f'train: {rel_train}')
                elif line.startswith('val:'):
                    new_lines.append(f'val: {rel_val}')
                elif line.startswith('test:'):
                    new_lines.append(f'test: {rel_test}')
                else:
                    new_lines.append(line)
            
            # 写回文件
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            
            print(f"✅ 配置文件已更新: {yaml_path}")
        else:
            print(f"⚠️  配置文件不存在: {yaml_path}")
            print("   请手动创建或检查路径")
    
    except Exception as e:
        print(f"⚠️  更新配置文件时出错: {e}")
        print("   你可能需要手动更新 traffic_signs.yaml 中的路径")
    
    print("\n" + "=" * 60)
    print("🎉 数据准备阶段全部完成！")
    print("=" * 60)
    print("\n现在你可以开始训练模型了：")
    print("   python step6_train_model.py")
    print("\n或者先测试单张图像的增强效果：")
    print("   python test_enhancement.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 增强过程中出现错误:")
    print("=" * 60)
    print(str(e))
    import traceback
    traceback.print_exc()
    print("\n请检查错误信息并修正后重试")
    sys.exit(1)

