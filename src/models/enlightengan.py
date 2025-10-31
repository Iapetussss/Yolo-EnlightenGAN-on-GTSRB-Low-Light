"""
EnlightenGAN 推理脚本
用于图像光照增强

基于 EnlightenGAN 论文: https://arxiv.org/abs/1906.06972
参考实现: https://github.com/arsenyinfo/EnlightenGAN-inference
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class EnlightenGANInference:
    """
    EnlightenGAN 推理类
    支持 ONNX 模型推理
    """
    
    def __init__(self, model_path='weights/enlightengan.onnx'):
        """
        初始化 EnlightenGAN 推理器
        
        Args:
            model_path: ONNX 模型路径
        """
        self.model_path = Path(model_path)
        self.session = None
        
        if self.model_path.exists():
            self.load_model()
        else:
            print(f"警告: 模型文件不存在: {model_path}")
            print("请从以下来源下载 EnlightenGAN ONNX 模型:")
            print("1. https://github.com/arsenyinfo/EnlightenGAN-inference")
            print("2. 或使用 PyTorch 版本并转换为 ONNX")
    
    def load_model(self):
        """
        加载 ONNX 模型
        """
        try:
            # 设置 ONNX Runtime 选项
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            
            print(f"EnlightenGAN 模型加载成功: {self.model_path}")
            print(f"使用设备: {self.session.get_providers()[0]}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.session = None
    
    def preprocess(self, image):
        """
        预处理输入图像
        
        Args:
            image: 输入图像 (BGR 格式)
            
        Returns:
            preprocessed: 预处理后的图像张量
        """
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [-1, 1]
        image_normalized = (image_rgb.astype(np.float32) / 127.5) - 1.0
        
        # 转置为 (C, H, W) 并添加 batch 维度
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_transposed, axis=0)
        
        return image_batch
    
    def postprocess(self, output):
        """
        后处理输出张量
        
        Args:
            output: 模型输出张量
            
        Returns:
            image: 后处理后的图像 (BGR 格式)
        """
        # 移除 batch 维度
        output = output.squeeze(0)
        
        # 转置为 (H, W, C)
        output = np.transpose(output, (1, 2, 0))
        
        # 反归一化到 [0, 255]
        output = ((output + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        # 转换为 BGR
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output_bgr
    
    def process(self, image):
        """
        处理图像 (主接口)
        
        Args:
            image: 输入图像 (BGR 格式)
            
        Returns:
            enhanced: 增强后的图像 (BGR 格式)
        """
        if self.session is None:
            print("警告: 模型未加载，使用传统方法增强")
            return self.fallback_enhancement(image)
        
        try:
            # 获取原始尺寸
            original_height, original_width = image.shape[:2]
            
            # 调整大小到模型输入尺寸 (可选，根据模型而定)
            # 某些 EnlightenGAN 模型要求特定尺寸
            input_size = (256, 256)  # 根据你的模型调整
            resized = cv2.resize(image, input_size)
            
            # 预处理
            input_tensor = self.preprocess(resized)
            
            # 推理
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            output_tensor = self.session.run([output_name], {input_name: input_tensor})[0]
            
            # 后处理
            enhanced = self.postprocess(output_tensor)
            
            # 调整回原始尺寸
            enhanced = cv2.resize(enhanced, (original_width, original_height))
            
            return enhanced
            
        except Exception as e:
            print(f"推理失败: {e}")
            return self.fallback_enhancement(image)
    
    def fallback_enhancement(self, image):
        """
        后备增强方法 (当模型不可用时)
        使用 CLAHE + Gamma 校正
        
        Args:
            image: 输入图像 (BGR 格式)
            
        Returns:
            enhanced: 增强后的图像
        """
        # 转换到 LAB 色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用 CLAHE 到 L 通道
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # 合并通道
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Gamma 校正
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced


def download_enlightengan_model(output_dir='weights'):
    """
    下载 EnlightenGAN ONNX 模型的说明
    """
    print("""
    EnlightenGAN 模型获取说明:
    
    方法 1: 使用预训练的 ONNX 模型
    --------------------------------
    1. 访问: https://github.com/arsenyinfo/EnlightenGAN-inference
    2. 下载 enlightengan.onnx 文件
    3. 将文件放到 weights/ 目录
    
    方法 2: 从 PyTorch 转换
    --------------------------------
    1. 克隆原始仓库: 
       git clone https://github.com/VITA-Group/EnlightenGAN
    2. 下载预训练权重
    3. 使用以下代码转换为 ONNX:
    
    import torch
    from models import create_model
    
    # 加载 PyTorch 模型
    model = create_model(opt)
    model.load_state_dict(torch.load('enlightengan.pth'))
    model.eval()
    
    # 转换为 ONNX
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy_input,
        'enlightengan.onnx',
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    
    方法 3: 使用后备方法
    --------------------------------
    如果无法获取 EnlightenGAN 模型，代码会自动使用传统的
    CLAHE + Gamma 校正方法进行图像增强，效果也不错！
    """)


# 使用示例
if __name__ == "__main__":
    # 显示模型下载说明
    download_enlightengan_model()
    
    # 测试推理
    print("\n测试 EnlightenGAN 推理...")
    
    # 创建推理器
    inference = EnlightenGANInference('weights/enlightengan.onnx')
    
    # 测试图像
    test_image_path = 'test_image.jpg'
    if Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        enhanced = inference.process(image)
        
        # 保存结果
        cv2.imwrite('enhanced_result.jpg', enhanced)
        print("增强结果已保存到 enhanced_result.jpg")
    else:
        print(f"测试图像不存在: {test_image_path}")

