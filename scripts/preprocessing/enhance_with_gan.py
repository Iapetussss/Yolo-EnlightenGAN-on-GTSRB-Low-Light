"""
步骤 2：使用 EnlightenGAN 批量增强数据集

将低光照数据集的所有图像使用 EnlightenGAN 增强
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import sys
import time

def check_prerequisites():
    """检查前置条件"""
    print("\n" + "=" * 60)
    print("检查前置条件...")
    print("=" * 60)
    
    # 检查模型
    model_path = Path('weights/enlightengan.onnx')
    if not model_path.exists():
        print("❌ 未找到 EnlightenGAN 模型")
        print("\n请先运行:")
        print("   python step_enlightengan_1_test.py")
        print("或")
        print("   python download_enlightengan_onnx.py")
        return False
    
    print(f"✅ 模型文件存在: {model_path}")
    
    # 检查低光照数据集
    possible_dirs = [
        'lowlight_images',
        'traffic_sign_data/lowlight_images'
    ]
    
    lowlight_dir = None
    for dir_path in possible_dirs:
        if Path(dir_path).exists():
            lowlight_dir = Path(dir_path)
            break
    
    if lowlight_dir is None:
        print("❌ 未找到低光照数据集")
        print("\n请先运行:")
        print("   python step4_create_lowlight.py")
        return False
    
    print(f"✅ 低光照数据集: {lowlight_dir}")
    
    return True, lowlight_dir

def load_enlightengan_model():
    """加载 EnlightenGAN 模型"""
    print("\n" + "=" * 60)
    print("加载 EnlightenGAN 模型...")
    print("=" * 60)
    
    try:
        from enlightengan_inference import EnlightenGANInference
        
        model = EnlightenGANInference('weights/enlightengan.onnx')
        
        if model.session is not None:
            print("✅ 模型加载成功")
            print(f"   设备: {model.session.get_providers()[0]}")
            return model
        else:
            print("❌ 模型加载失败")
            return None
            
    except Exception as e:
        print(f"❌ 加载错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_image_files(directory):
    """获取目录下所有图像文件"""
    extensions = ['.jpg', '.jpeg', '.png', '.ppm']
    image_files = []
    
    for ext in extensions:
        image_files.extend(list(directory.rglob(f'*{ext}')))
    
    return image_files

def enhance_dataset(model, input_dir, output_dir):
    """批量增强数据集"""
    print("\n" + "=" * 60)
    print(f"增强数据集: {input_dir}")
    print("=" * 60)
    
    # 获取所有图像
    image_files = get_image_files(input_dir)
    total = len(image_files)
    
    if total == 0:
        print("❌ 未找到图像文件")
        return False
    
    print(f"找到 {total} 张图像")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    success_count = 0
    fail_count = 0
    total_time = 0
    
    print("\n开始增强...")
    
    # 处理每张图像
    for img_path in tqdm(image_files, desc="增强进度"):
        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                fail_count += 1
                continue
            
            # 增强
            start = time.time()
            enhanced = model.process(image)
            elapsed = time.time() - start
            total_time += elapsed
            
            # 构建输出路径（保持目录结构）
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            
            # 创建输出目录
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存
            cv2.imwrite(str(out_path), enhanced)
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            tqdm.write(f"处理失败 {img_path}: {e}")
    
    # 统计
    avg_time = (total_time / success_count * 1000) if success_count > 0 else 0
    
    print("\n" + "=" * 60)
    print("增强完成")
    print("=" * 60)
    print(f"✅ 成功: {success_count}/{total}")
    print(f"❌ 失败: {fail_count}/{total}")
    print(f"⏱️  平均处理时间: {avg_time:.1f} ms/张")
    print(f"📊 总处理时间: {total_time/60:.1f} 分钟")
    
    return success_count > 0

def copy_labels(input_dir, output_dir):
    """复制标签文件"""
    print("\n" + "=" * 60)
    print("复制标签文件...")
    print("=" * 60)
    
    # 找到标签目录
    # 假设标签在 yolo_dataset/labels/ 或类似位置
    possible_label_dirs = [
        Path('yolo_dataset/labels'),
        Path('traffic_sign_data/labels'),
        input_dir.parent / 'labels'
    ]
    
    label_dir = None
    for dir_path in possible_label_dirs:
        if dir_path.exists():
            label_dir = dir_path
            break
    
    if label_dir is None:
        print("⚠️  未找到标签目录，跳过")
        print("   请手动复制标签文件")
        return False
    
    print(f"标签目录: {label_dir}")
    
    # 获取所有标签文件
    label_files = list(label_dir.rglob('*.txt'))
    
    if not label_files:
        print("⚠️  未找到标签文件")
        return False
    
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 创建输出标签目录
    output_label_dir = output_dir.parent / 'labels'
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制标签
    copied = 0
    for label_path in tqdm(label_files, desc="复制标签"):
        try:
            rel_path = label_path.relative_to(label_dir)
            out_path = output_label_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, out_path)
            copied += 1
        except Exception as e:
            tqdm.write(f"复制失败 {label_path}: {e}")
    
    print(f"✅ 复制完成: {copied}/{len(label_files)}")
    
    return True

def create_summary(output_base_dir):
    """创建摘要文件"""
    summary_path = output_base_dir / 'enhancement_summary.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EnlightenGAN 数据集增强摘要\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"增强方法: EnlightenGAN (ONNX)\n")
        f.write(f"输出目录: {output_base_dir}\n\n")
        
        # 统计各集图像数
        for split in ['train', 'val', 'test']:
            split_dir = output_base_dir / split
            if split_dir.exists():
                image_count = len(list(split_dir.rglob('*.png'))) + len(list(split_dir.rglob('*.jpg')))
                f.write(f"{split.upper()} 集: {image_count} 张图像\n")
        
        f.write("\n下一步:\n")
        f.write("  python step_enlightengan_3_reorganize.py\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n📄 摘要已保存: {summary_path}")

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  EnlightenGAN 批量数据集增强".center(70))
    print("=" * 70)
    
    # 1. 检查前置条件
    check_result = check_prerequisites()
    if not check_result:
        sys.exit(1)
    
    _, lowlight_dir = check_result
    
    # 2. 加载模型
    model = load_enlightengan_model()
    if model is None:
        print("\n❌ 无法加载模型，终止")
        sys.exit(1)
    
    # 3. 确认开始
    print("\n" + "=" * 70)
    print("⚠️  注意事项:")
    print("=" * 70)
    print("1. 这个过程会处理所有图像（约 50,000 张）")
    print("2. 预计时间:")
    print("   - GPU: 1.5-2 小时")
    print("   - CPU: 4-6 小时")
    print("3. 输出目录: enlightengan_enhanced/")
    print("4. 确保有足够磁盘空间（约 5GB）")
    
    response = input("\n是否继续？(y/N): ").strip().lower()
    if response != 'y':
        print("已取消")
        sys.exit(0)
    
    # 4. 增强各个数据集
    output_base = Path('enlightengan_enhanced')
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_input = lowlight_dir / split
        split_output = output_base / split
        
        if not split_input.exists():
            print(f"\n⚠️  跳过 {split} 集（目录不存在）")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"处理 {split.upper()} 集")
        print(f"{'=' * 70}")
        
        success = enhance_dataset(model, split_input, split_output)
        
        if not success:
            print(f"❌ {split} 集增强失败")
    
    # 5. 复制标签
    # copy_labels(lowlight_dir, output_base)
    
    # 6. 创建摘要
    create_summary(output_base)
    
    # 7. 完成
    print("\n" + "=" * 70)
    print("  ✅ 所有数据集增强完成！".center(70))
    print("=" * 70)
    
    print("\n📁 增强后的数据集:")
    print(f"   {output_base.absolute()}")
    
    print("\n📊 数据统计:")
    for split in splits:
        split_dir = output_base / split
        if split_dir.exists():
            count = len(list(split_dir.rglob('*.png'))) + len(list(split_dir.rglob('*.jpg')))
            print(f"   {split.upper():<6}: {count:>6} 张")
    
    print("\n🎯 下一步:")
    print("   1. 重组数据集为 YOLO 格式:")
    print("      python step_enlightengan_3_reorganize.py")
    print("\n   2. 或者直接查看增强效果:")
    print("      打开 enlightengan_enhanced/ 目录查看图像")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    main()

