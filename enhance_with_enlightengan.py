"""
EnlightenGAN图像增强脚本
用于增强低光照交通标志图像数据集
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

print("=" * 60)
print("✨ 使用EnlightenGAN增强低光照图像")
print("=" * 60)

# 检查Python版本
if sys.version_info < (3, 8):
    print("❌ Python版本过低，请使用Python 3.8+")
    sys.exit(1)

# 设置数据集路径
INPUT_DIR = Path("traffic_sign_data/low_light")
OUTPUT_DIR = Path("traffic_sign_data_enhanced")

# 检查输入目录
if not INPUT_DIR.exists():
    print(f"❌ 低光照数据集不存在: {INPUT_DIR}")
    print("请确保traffic_sign_data/low_light目录下有images和labels文件夹")
    sys.exit(1)

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 创建输出目录结构
for split in ['train', 'val', 'test']:
    (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

print(f"\n输入路径: {INPUT_DIR}")
print(f"输出路径: {OUTPUT_DIR}")

# 定义增强方法选择
def choose_enhancement_method():
    """选择图像增强方法"""
    print("\n" + "=" * 60)
    print("选择增强方法:")
    print("=" * 60)
    print("1. 使用EnlightenGAN增强 (需要下载模型)")
    print("2. 使用传统方法增强 (CLAHE + Gamma校正，无需额外模型)")
    
    choice = input("\n请选择 [1/2]: ").strip()
    
    if choice == '1':
        # 检查EnlightenGAN模型
        model_path = Path("weights/enlightengan.onnx")
        if not model_path.exists():
            print("\n❌ 未找到EnlightenGAN模型")
            print("正在创建下载模型的脚本...")
            create_download_script()
            print("请运行: python download_enlightengan.py 下载模型")
            print("\n将自动使用传统方法增强...")
            return "traditional"
        return "enlightengan"
    else:
        return "traditional"

def create_download_script():
    """创建下载EnlightenGAN模型的脚本"""
    script_content = '''"""
下载EnlightenGAN ONNX模型的脚本
"""

import os
import sys
import requests
from tqdm import tqdm
from pathlib import Path

print("=" * 60)
print("📥 下载EnlightenGAN ONNX模型")
print("=" * 60)

# 创建weights目录
weights_dir = Path("weights")
weights_dir.mkdir(exist_ok=True)

# 模型URL (这里使用一个示例URL，实际需要替换为真实的下载链接)
MODEL_URL = "https://github.com/VITA-Group/EnlightenGAN/releases/download/v1.0/enlightengan.onnx"
MODEL_PATH = weights_dir / "enlightengan.onnx"

# 下载模型
def download_model():
    try:
        print(f"开始下载模型: {MODEL_URL}")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        
        with open(MODEL_PATH, 'wb') as file, tqdm(
            desc=str(MODEL_PATH),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size=block_size):
                size = file.write(data)
                bar.update(size)
        
        print(f"\n✅ 模型下载成功: {MODEL_PATH}")
        print("现在可以使用EnlightenGAN增强图像了")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n提示：由于网络限制，您可能需要手动下载模型并放在weights目录下")
        print("模型名称应为: enlightengan.onnx")

if __name__ == "__main__":
    download_model()
'''
    
    with open("download_enlightengan.py", "w", encoding="utf-8") as f:
        f.write(script_content)

def traditional_enhancement(image):
    """传统图像增强方法 (CLAHE + Gamma校正)"""
    # 转换到LAB色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 应用CLAHE到L通道
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Gamma校正
    gamma = 1.2
    l_gamma = np.array(255 * (l_clahe / 255) ** (1 / gamma), dtype='uint8')
    
    # 合并通道
    enhanced_lab = cv2.merge([l_gamma, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def enhance_with_enlightengan(image):
    """使用EnlightenGAN增强图像"""
    # 这里需要实现EnlightenGAN的推理代码
    # 由于模型可能不存在，我们先尝试导入，如果失败则使用传统方法
    try:
        import onnxruntime as ort
        
        # 加载ONNX模型
        model_path = "weights/enlightengan.onnx"
        session = ort.InferenceSession(model_path)
        
        # 预处理图像
        input_name = session.get_inputs()[0].name
        img = cv2.resize(image, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # 推理
        outputs = session.run(None, {input_name: img})
        
        # 后处理
        enhanced = outputs[0][0]
        enhanced = np.transpose(enhanced, (1, 2, 0))
        enhanced = (enhanced * 255.0).astype(np.uint8)
        enhanced = cv2.resize(enhanced, (image.shape[1], image.shape[0]))
        
        return enhanced
    except Exception as e:
        print(f"⚠️ EnlightenGAN推理失败: {e}")
        print("回退到传统方法增强")
        return traditional_enhancement(image)

def copy_labels(input_labels_dir, output_labels_dir):
    """复制标签文件"""
    if input_labels_dir.exists():
        for label_file in input_labels_dir.glob("*.txt"):
            shutil.copy2(label_file, output_labels_dir / label_file.name)
        print(f"✅ 已复制 {len(list(input_labels_dir.glob('*.txt')))} 个标签文件")

def process_split(split, method):
    """处理单个数据集分割"""
    print(f"\n" + "=" * 60)
    print(f"处理 {split.upper()} 集...")
    print("=" * 60)
    
    input_images_dir = INPUT_DIR / 'images' / split
    input_labels_dir = INPUT_DIR / 'labels' / split
    output_images_dir = OUTPUT_DIR / 'images' / split
    output_labels_dir = OUTPUT_DIR / 'labels' / split
    
    # 确保目录存在
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.ppm']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_images_dir.glob(f"*{ext}")))
    
    if not image_files:
        print(f"❌ 未找到 {split} 图像文件")
        return 0
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 处理每张图像
    for i, img_path in enumerate(tqdm(image_files), 1):
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️  无法读取图像: {img_path}")
            continue
        
        # 增强图像
        if method == "enlightengan":
            enhanced = enhance_with_enlightengan(image)
        else:
            enhanced = traditional_enhancement(image)
        
        # 保存增强后的图像
        output_path = output_images_dir / img_path.name
        cv2.imwrite(str(output_path), enhanced)
    
    # 复制标签文件
    copy_labels(input_labels_dir, output_labels_dir)
    
    return len(image_files)

def create_dataset_config():
    """创建YOLO数据集配置文件"""
    config_path = OUTPUT_DIR / 'traffic_signs_enhanced.yaml'
    
    config_content = f'''
path: {OUTPUT_DIR.absolute()}
train: images/train
val: images/val
test: images/test

# 类别数和名称
nc: 43
names: {list(range(43))}
'''
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\n✅ 已创建数据集配置文件: {config_path}")
    return config_path

def main():
    """主函数"""
    # 选择增强方法
    method = choose_enhancement_method()
    print(f"\n使用增强方法: {'EnlightenGAN' if method == 'enlightengan' else '传统方法 (CLAHE + Gamma校正)'}")
    
    # 开始增强
    total_processed = 0
    
    # 处理各个数据集分割
    for split in ['train', 'val', 'test']:
        count = process_split(split, method)
        total_processed += count
    
    # 创建数据集配置
    config_path = create_dataset_config()
    
    print(f"\n" + "=" * 60)
    print(f"🎉 图像增强完成!")
    print(f"总共处理: {total_processed} 张图像")
    print(f"增强方法: {'EnlightenGAN' if method == 'enlightengan' else '传统方法'}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"配置文件: {config_path}")
    print("\n下一步:")
    print("1. 使用增强后的数据集运行YOLO检测:")
    print("   python detect_with_yolo.py")
    print("2. 比较增强前后的检测效果:")
    print("   python compare_detection_results.py")
    print("=" * 60)

if __name__ == "__main__":
    main()