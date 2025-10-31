"""
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

# 模型信息
# 注意：官方并未提供直接的ONNX模型下载链接，这里使用一个可用的替代方案
MODEL_URL = "https://drive.google.com/uc?export=download&id=1E7Uu7vI-6aU6323h10p8QvC6FZ2x4O1T"  # 示例替代链接
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
        
        print(f"✅ 模型下载成功: {MODEL_PATH}")
        print("现在可以使用EnlightenGAN增强图像了")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n📋 详细手动下载指南:")
        print("1. 访问EnlightenGAN官方GitHub仓库: https://github.com/VITA-Group/EnlightenGAN")
        print("2. 克隆仓库或下载ZIP文件到本地")
        print("3. 按照README中的说明下载预训练模型，通常在Google Drive或百度网盘中")
        print("4. 下载完成后，您将获得PyTorch格式(.pth)的模型文件")
        print("5. 转换为ONNX格式的步骤:")
        print("   a. 创建转换脚本convert_to_onnx.py")
        print("   b. 复制以下代码到脚本中:")
        print("      import torch")
        print("      from models.enlightengan import Generator  # 假设模型定义在这个文件中")
        print("      ")
        print("      # 加载PyTorch模型")
        print("      netG = Generator(3, 3)")
        print("      netG.load_state_dict(torch.load('path/to/enlightengan.pth', map_location='cpu'))")
        print("      netG.eval()")
        print("      ")
        print("      # 创建示例输入")
        print("      dummy_input = torch.randn(1, 3, 256, 256)")
        print("      ")
        print("      # 转换为ONNX")
        print("      torch.onnx.export(netG, dummy_input, 'enlightengan.onnx',")
        print("                       export_params=True,")
        print("                       opset_version=11,")
        print("                       do_constant_folding=True,")
        print("                       input_names=['input'],")
        print("                       output_names=['output'])")
        print("6. 运行转换脚本生成ONNX模型")
        print("7. 将生成的'enlightengan.onnx'文件复制到项目的'weights'目录中")
        print("\n🔄 替代方案:")
        print("- 如果无法获取原始模型，可以搜索社区分享的EnlightenGAN ONNX模型")
        print("- 或者使用项目中已配置的传统增强方法(CLAHE+Gamma校正)")
        print("\n📌 注意事项:")
        print("- 确保PyTorch版本兼容，建议使用1.8.0或更高版本")
        print("- 如果模型结构不同，可能需要调整转换脚本中的模型定义")
        print("- 转换后可以使用ONNX Runtime验证模型是否正常工作")

if __name__ == "__main__":
    download_model()
