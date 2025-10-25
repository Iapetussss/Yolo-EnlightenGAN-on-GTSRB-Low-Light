"""
低光照交通标志检测项目
使用 EnlightenGAN + YOLOv8 实现 GTSRB 数据集的交通标志检测

参考项目: Tiger-Detection-using-EnlightenGAN-and-Yolo
数据集: GTSRB (German Traffic Sign Recognition Benchmark)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image
import torch


class GTSRBEnlightenGANDetector:
    """
    GTSRB 交通标志检测器
    结合 EnlightenGAN 图像增强和 YOLOv8 目标检测
    """
    
    def __init__(self, config_path='../traffic_signs.yaml'):
        """
        初始化检测器
        
        Args:
            config_path: YOLOv8 配置文件路径
        """
        self.config_path = config_path
        self.enlighten_model = None
        self.yolo_model = None
        
    def setup_enlightengan(self, model_path='weights/enlightengan.pth'):
        """
        设置 EnlightenGAN 模型
        
        Args:
            model_path: EnlightenGAN 预训练模型路径
        """
        print("正在加载 EnlightenGAN 模型...")
        try:
            # 这里需要 EnlightenGAN 的推理代码
            # 可以使用 ONNX 版本或原始 PyTorch 版本
            from enlightengan_inference import EnlightenGANInference
            self.enlighten_model = EnlightenGANInference(model_path)
            print("EnlightenGAN 模型加载成功！")
        except Exception as e:
            print(f"警告: EnlightenGAN 模型加载失败: {e}")
            print("将使用传统图像增强方法作为后备方案")
            
    def setup_yolov8(self, model_path='yolov8n.pt'):
        """
        设置 YOLOv8 模型
        
        Args:
            model_path: YOLOv8 预训练模型路径或检查点
        """
        print("正在加载 YOLOv8 模型...")
        from ultralytics import YOLO
        self.yolo_model = YOLO(model_path)
        print("YOLOv8 模型加载成功！")
        
    def enhance_image(self, image_path, output_path=None, method='enlightengan'):
        """
        增强图像光照
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            method: 增强方法 ('enlightengan' 或 'traditional')
            
        Returns:
            enhanced_image: 增强后的图像
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        if method == 'enlightengan' and self.enlighten_model is not None:
            # 使用 EnlightenGAN 增强
            enhanced = self.enlighten_model.process(image)
        else:
            # 使用传统方法增强（CLAHE + Gamma 校正）
            enhanced = self.traditional_enhancement(image)
            
        # 保存增强后的图像
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(str(output_path), enhanced)
            
        return enhanced
    
    def traditional_enhancement(self, image):
        """
        传统图像增强方法（作为 EnlightenGAN 的后备方案）
        使用 CLAHE (对比度限制自适应直方图均衡) + Gamma 校正
        
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
    
    def enhance_dataset(self, input_dir, output_dir, method='enlightengan'):
        """
        批量增强数据集图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            method: 增强方法
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.ppm']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.rglob(f'*{ext}')))
            
        print(f"找到 {len(image_files)} 张图像，开始增强...")
        
        for img_file in tqdm(image_files, desc="增强图像"):
            # 构建输出路径，保持目录结构
            rel_path = img_file.relative_to(input_path)
            out_file = output_path / rel_path
            
            try:
                self.enhance_image(img_file, out_file, method=method)
            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {e}")
                
        print("图像增强完成！")
    
    def train_yolov8(self, epochs=100, imgsz=640, batch=16, device='0'):
        """
        训练 YOLOv8 模型
        
        Args:
            epochs: 训练轮数
            imgsz: 图像尺寸
            batch: 批次大小
            device: 设备 ('0' for GPU, 'cpu' for CPU)
        """
        if self.yolo_model is None:
            raise ValueError("请先使用 setup_yolov8() 加载模型")
            
        print("开始训练 YOLOv8 模型...")
        results = self.yolo_model.train(
            data=self.config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project='runs/train',
            name='gtsrb_enlightengan',
            plots=True,
            save=True,
            verbose=True
        )
        
        print("训练完成！")
        return results
    
    def validate(self, split='val', device='0'):
        """
        验证模型
        
        Args:
            split: 数据集划分 ('val' 或 'test')
            device: 设备
        """
        if self.yolo_model is None:
            raise ValueError("请先使用 setup_yolov8() 加载模型")
            
        print(f"开始验证模型 (数据集: {split})...")
        results = self.yolo_model.val(
            data=self.config_path,
            split=split,
            device=device,
            plots=True
        )
        
        return results
    
    def predict(self, image_path, conf=0.25, save=True, save_dir='runs/predict'):
        """
        预测单张图像
        
        Args:
            image_path: 图像路径
            conf: 置信度阈值
            save: 是否保存结果
            save_dir: 保存目录
            
        Returns:
            results: 预测结果
        """
        if self.yolo_model is None:
            raise ValueError("请先使用 setup_yolov8() 加载模型")
            
        results = self.yolo_model.predict(
            source=image_path,
            conf=conf,
            save=save,
            project=save_dir,
            verbose=False
        )
        
        return results
    
    def visualize_results(self, image_path, results, figsize=(15, 5)):
        """
        可视化检测结果
        
        Args:
            image_path: 原始图像路径
            results: YOLOv8 预测结果
            figsize: 图像大小
        """
        # 读取原始图像
        original = cv2.imread(str(image_path))
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # 增强图像
        enhanced = self.enhance_image(image_path, method='traditional')
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        # 获取标注后的图像
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # 绘制对比图
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].imshow(original_rgb)
        axes[0].set_title('原始图像 (低光照)')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_rgb)
        axes[1].set_title('增强后图像')
        axes[1].axis('off')
        
        axes[2].imshow(annotated_rgb)
        axes[2].set_title('检测结果')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig


def create_low_light_dataset(input_dir, output_dir, gamma_range=(0.3, 0.7)):
    """
    创建低光照数据集
    通过调整 Gamma 值来模拟低光照环境
    
    Args:
        input_dir: 原始 GTSRB 数据集目录
        output_dir: 输出低光照数据集目录
        gamma_range: Gamma 值范围 (越小越暗)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像
    image_files = list(input_path.rglob('*.ppm')) + list(input_path.rglob('*.jpg'))
    
    print(f"创建低光照数据集: {len(image_files)} 张图像")
    
    for img_file in tqdm(image_files, desc="生成低光照图像"):
        # 读取图像
        image = cv2.imread(str(img_file))
        if image is None:
            continue
            
        # 随机选择 gamma 值
        gamma = np.random.uniform(*gamma_range)
        
        # 应用 gamma 变换
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        low_light = cv2.LUT(image, table)
        
        # 保存
        rel_path = img_file.relative_to(input_path)
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), low_light)
    
    print("低光照数据集创建完成！")


# 使用示例
if __name__ == "__main__":
    # 创建检测器实例
    detector = GTSRBEnlightenGANDetector(config_path='../traffic_signs.yaml')
    
    # 步骤 1: 设置 YOLOv8 模型
    print("\n=== 步骤 1: 加载 YOLOv8 模型 ===")
    detector.setup_yolov8('yolov8n.pt')  # 使用 nano 版本，可改为 yolov8s.pt, yolov8m.pt 等
    
    # 步骤 2: (可选) 创建低光照数据集
    # print("\n=== 步骤 2: 创建低光照数据集 ===")
    # create_low_light_dataset('path/to/gtsrb', 'path/to/gtsrb_low_light')
    
    # 步骤 3: 增强图像
    # print("\n=== 步骤 3: 图像增强 ===")
    # detector.enhance_dataset(
    #     input_dir='path/to/gtsrb_low_light/train',
    #     output_dir='traffic_sign_data/enhanced_images/train',
    #     method='traditional'
    # )
    
    # 步骤 4: 训练模型
    # print("\n=== 步骤 4: 训练 YOLOv8 ===")
    # detector.train_yolov8(epochs=100, imgsz=640, batch=16, device='0')
    
    # 步骤 5: 验证模型
    # print("\n=== 步骤 5: 验证模型 ===")
    # detector.validate(split='val')
    
    # 步骤 6: 测试单张图像
    # print("\n=== 步骤 6: 测试预测 ===")
    # results = detector.predict('path/to/test_image.jpg')
    # detector.visualize_results('path/to/test_image.jpg', results)
    
    print("\n项目初始化完成！请根据需要取消注释相应步骤。")

