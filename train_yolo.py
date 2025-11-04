import os
import argparse
from ultralytics import YOLO
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='训练YOLOv8模型用于交通标志检测')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='预训练模型路径或名称')
    parser.add_argument('--data', type=str, default='configs/new_gtsrb.yaml', help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像大小')
    parser.add_argument('--name', type=str, default='new_gtsrb_yolov8', help='实验名称')
    parser.add_argument('--device', type=str, default='0', help='设备选择，0表示GPU，-1表示CPU')
    return parser.parse_args()

def train_model(args):
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != '-1' else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    model = YOLO(args.model)
    print(f'已加载模型: {args.model}')
    
    # 创建保存结果的目录
    if not os.path.exists('runs/detect'):
        os.makedirs('runs/detect')
    
    print('开始训练...')
    # 训练模型
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        name=args.name,
        device=args.device,
        workers=0,  # Windows系统推荐设置为0
        pretrained=True,
        optimizer='SGD',
        cos_lr=True,
        resume=False
    )
    
    print('训练完成！')
    print(f'模型保存路径: runs/detect/{args.name}')
    
    # 评估模型
    print('开始评估模型...')
    metrics = model.val()
    print(f'mAP50: {metrics.box.map}')
    print(f'mAP50-95: {metrics.box.map50_95}')
    
    # 导出模型为ONNX格式
    try:
        print('导出模型为ONNX格式...')
        model.export(format='onnx')
        print('模型导出成功！')
    except Exception as e:
        print(f'模型导出失败: {e}')

if __name__ == '__main__':
    args = parse_arguments()
    train_model(args)