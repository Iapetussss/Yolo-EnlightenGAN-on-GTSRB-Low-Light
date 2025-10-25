"""
步骤 5 (EnlightenGAN 版): 增强低光照图像
使用 EnlightenGAN 或改进的传统方法增强图像
"""

import sys
from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm

print("=" * 60)
print("✨ 步骤 5: 增强低光照图像 (EnlightenGAN 版)")
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

# 检查增强方法
print("\n" + "=" * 60)
print("选择增强方法:")
print("=" * 60)

# 检查是否有 EnlightenGAN 模型
weights_dir = Path(__file__).parent / 'weights'
onnx_model = weights_dir / "enlightengan.onnx"
pth_model = weights_dir / "enlightengan.pth"
use_traditional_flag = Path(__file__).parent / 'use_traditional_enhanced.txt'

has_onnx = onnx_model.exists()
has_pth = pth_model.exists()
use_traditional = use_traditional_flag.exists()

if has_onnx:
    print(f"✅ 找到 ONNX 模型: {onnx_model}")
    method_choice = "onnx"
elif has_pth:
    print(f"✅ 找到 PyTorch 模型: {pth_model}")
    method_choice = "pytorch"
elif use_traditional:
    print("✅ 使用改进的传统方法")
    method_choice = "enhanced_traditional"
else:
    print("⚠️  未找到 EnlightenGAN 模型")
    print("\n可用选项:")
    print("1. 使用改进的传统方法（推荐）")
    print("2. 下载 EnlightenGAN 模型")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == '2':
        print("\n请先运行: python download_enlightengan_model.py")
        sys.exit(0)
    else:
        method_choice = "enhanced_traditional"

print(f"\n使用方法: {method_choice}")

# 定义增强函数
def enhanced_traditional_method(image):
    """
    改进的传统方法
    结合 CLAHE, Gamma 校正, 和 Multi-Scale Retinex
    """
    # 1. 转换到 LAB 色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. 应用 CLAHE 到 L 通道
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # 3. Gamma 校正
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    l_gamma = cv2.LUT(l_clahe, table)
    
    # 4. Multi-Scale Retinex (简化版)
    # 高斯模糊
    gaussian = cv2.GaussianBlur(l_gamma, (0, 0), 15)
    
    # Retinex: log(image) - log(gaussian)
    l_float = l_gamma.astype(np.float32) + 1.0
    gaussian_float = gaussian.astype(np.float32) + 1.0
    
    retinex = np.log(l_float) - np.log(gaussian_float)
    
    # 归一化到 0-255
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    l_retinex = retinex.astype(np.uint8)
    
    # 5. 合并通道
    enhanced_lab = cv2.merge([l_retinex, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 6. 颜色增强
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 增加饱和度
    s = cv2.add(s, 10)
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    enhanced_hsv = cv2.merge([h, s, v])
    final = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    return final

def simple_traditional_method(image):
    """
    简单的传统方法（后备方案）
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    
    return enhanced

def enlightengan_onnx_method(image, model_path):
    """
    使用 ONNX 版本的 EnlightenGAN
    """
    try:
        import onnxruntime as ort
        
        # 预处理
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb.astype(np.float32) / 127.5) - 1.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # 推理
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: img_batch})[0]
        
        # 后处理
        output = output.squeeze(0)
        output = np.transpose(output, (1, 2, 0))
        output = ((output + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output_bgr
        
    except Exception as e:
        print(f"\n⚠️  ONNX 推理失败: {e}")
        print("   回退到传统方法")
        return enhanced_traditional_method(image)

# 选择增强方法
if method_choice == "onnx":
    print("\n正在加载 ONNX 模型...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_model))
        print("✅ ONNX 模型加载成功")
        enhance_func = lambda img: enlightengan_onnx_method(img, onnx_model)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("回退到改进的传统方法")
        enhance_func = enhanced_traditional_method
elif method_choice == "pytorch":
    print("\n⚠️  PyTorch 模型需要额外配置")
    print("暂时使用改进的传统方法")
    enhance_func = enhanced_traditional_method
else:
    enhance_func = enhanced_traditional_method

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

def enhance_dataset(input_dir, output_dir, enhance_func):
    """批量增强图像"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像
    image_files = list(input_path.glob('*.png'))
    
    if not image_files:
        print(f"   ⚠️  未找到图像: {input_path}")
        return 0
    
    for img_file in tqdm(image_files, desc=f"   增强 {input_path.name}"):
        try:
            # 读取图像
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # 增强
            enhanced = enhance_func(image)
            
            # 保存
            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), enhanced)
            
        except Exception as e:
            print(f"\n   ⚠️  处理 {img_file.name} 时出错: {e}")
            continue
    
    return len(image_files)

try:
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        input_images = input_root / 'images' / split
        output_images = output_root / split
        
        if not input_images.exists():
            print(f"\n⚠️  跳过 {split} (目录不存在)")
            continue
        
        print(f"\n增强 {split} 集...")
        count = enhance_dataset(input_images, output_images, enhance_func)
        total_images += count
        print(f"✅ {split} 集完成: {count} 张图像")
        
        # 复制标注文件
        src_labels = input_root / 'labels' / split
        dst_labels = output_root.parent / 'labels' / split
        dst_labels.mkdir(parents=True, exist_ok=True)
        
        if src_labels.exists():
            label_files = list(src_labels.glob('*.txt'))
            for label_file in label_files:
                shutil.copy(str(label_file), str(dst_labels / label_file.name))
            print(f"✅ 复制 {len(label_files)} 个标注文件")
    
    print("\n" + "=" * 60)
    print("✅ 图像增强完成！")
    print("=" * 60)
    print(f"\n总共增强: {total_images} 张图像")
    print(f"输出位置: {output_root}")
    
    # 保存输出路径
    output_config = Path(__file__).parent / 'enhanced_dataset_path.txt'
    with open(output_config, 'w') as f:
        f.write(str(output_root.absolute()))
    
    # 更新 YAML 配置
    print("\n" + "=" * 60)
    print("更新 YOLOv8 配置文件...")
    print("=" * 60)
    
    yaml_path = Path(__file__).parent.parent / 'traffic_signs.yaml'
    
    try:
        rel_train = output_root / 'train'
        rel_val = output_root / 'val'
        rel_test = output_root / 'test'
        
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            backup_path = yaml_path.parent / 'traffic_signs.yaml.backup'
            shutil.copy(yaml_path, backup_path)
            
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
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            
            print(f"✅ 配置文件已更新: {yaml_path}")
        else:
            print(f"⚠️  配置文件不存在: {yaml_path}")
    
    except Exception as e:
        print(f"⚠️  更新配置文件时出错: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 数据准备阶段全部完成！")
    print("=" * 60)
    print("\n现在你可以开始训练模型了：")
    print("   python step6_train_model.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 增强过程中出现错误:")
    print("=" * 60)
    print(str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

