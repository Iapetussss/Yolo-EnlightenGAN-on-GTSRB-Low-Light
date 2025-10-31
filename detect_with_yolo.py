"""
YOLOv8目标检测脚本
用于对低光照和增强后的交通标志图像进行检测
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json

print("=" * 60)
print("🚦 使用YOLOv8进行交通标志检测")
print("=" * 60)

# 设置路径
LOWLIGHT_DIR = Path("traffic_sign_data/low_light")
ENHANCED_DIR = Path("traffic_sign_data_enhanced")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 检查目录
if not LOWLIGHT_DIR.exists():
    print(f"❌ 低光照数据集不存在: {LOWLIGHT_DIR}")
    sys.exit(1)

def setup_yolo_model():
    """设置YOLOv8模型"""
    print("\n" + "=" * 60)
    print("加载YOLOv8模型...")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        # 选择模型
        print("可用的YOLOv8模型:")
        print("1. yolov8n.pt (nano, 最快)")
        print("2. yolov8s.pt (small, 平衡速度和精度)")
        print("3. yolov8m.pt (medium, 高精度)")
        print("4. 自定义模型路径")
        
        choice = input("\n请选择 [1-4]: ").strip()
        
        if choice == '1':
            model_path = 'yolov8n.pt'
        elif choice == '2':
            model_path = 'yolov8s.pt'
        elif choice == '3':
            model_path = 'yolov8m.pt'
        elif choice == '4':
            model_path = input("请输入自定义模型路径: ").strip()
            if not model_path:
                model_path = 'yolov8n.pt'
                print("使用默认模型: yolov8n.pt")
        else:
            model_path = 'yolov8n.pt'
            print("使用默认模型: yolov8n.pt")
        
        print(f"\n加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 设置设备
        device = input("\n使用设备 (0/cpu，默认 0): ").strip()
        if not device:
            device = '0'
        
        print(f"使用设备: {device}")
        
        # 设置置信度阈值
        conf_thres = input("\n检测置信度阈值 (0-1，默认 0.25): ").strip()
        try:
            conf_thres = float(conf_thres) if conf_thres else 0.25
            if not (0 <= conf_thres <= 1):
                conf_thres = 0.25
        except:
            conf_thres = 0.25
        
        print(f"置信度阈值: {conf_thres}")
        
        return model, device, conf_thres
        
    except Exception as e:
        print(f"❌ 加载YOLOv8模型失败: {e}")
        sys.exit(1)

def detect_dataset(model, dataset_dir, output_dir, device, conf_thres, split="test"):
    """对数据集进行检测"""
    print(f"\n" + "=" * 60)
    print(f"对 {split} 集进行检测: {dataset_dir}")
    print("=" * 60)
    
    images_dir = dataset_dir / 'images' / split
    if not images_dir.exists():
        print(f"❌ 图像目录不存在: {images_dir}")
        return None
    
    # 创建输出目录
    detect_output_dir = output_dir / split
    detect_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.ppm']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    if not image_files:
        print(f"❌ 未找到 {split} 图像文件")
        return None
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 存储检测结果
    detection_results = []
    
    # 检测每张图像
    for img_path in tqdm(image_files):
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️  无法读取图像: {img_path}")
            continue
        
        # 运行检测
        results = model(image, device=device, conf=conf_thres, save=False)
        
        # 处理结果
        detections = []
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # 保存检测结果
        result = {
            'image_path': str(img_path),
            'filename': img_path.name,
            'detections': detections,
            'num_detections': len(detections)
        }
        detection_results.append(result)
        
        # 保存可视化结果
        if len(detections) > 0:
            # 绘制边界框
            viz_img = image.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cls = det['class_id']
                conf = det['confidence']
                
                # 绘制边界框和标签
                cv2.rectangle(viz_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{cls}: {conf:.2f}"
                cv2.putText(viz_img, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存可视化图像
            viz_path = detect_output_dir / f"det_{img_path.name}"
            cv2.imwrite(str(viz_path), viz_img)
    
    # 保存检测结果到JSON文件
    json_path = output_dir / f"{split}_detections.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 检测完成，结果保存到: {json_path}")
    
    # 计算统计信息
    total_detections = sum([r['num_detections'] for r in detection_results])
    avg_detections = total_detections / len(detection_results) if detection_results else 0
    
    print(f"\n检测统计:")
    print(f"- 总图像数: {len(detection_results)}")
    print(f"- 总检测数: {total_detections}")
    print(f"- 平均每张图像检测数: {avg_detections:.2f}")
    
    return {
        'results': detection_results,
        'stats': {
            'total_images': len(detection_results),
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections
        }
    }

def main():
    """主函数"""
    # 设置YOLO模型
    model, device, conf_thres = setup_yolo_model()
    
    # 选择检测数据集
    print("\n" + "=" * 60)
    print("选择检测数据集:")
    print("=" * 60)
    print("1. 仅低光照数据集")
    print("2. 仅增强后数据集")
    print("3. 两者都检测 (推荐，用于对比)")
    
    choice = input("\n请选择 [1-3]: ").strip()
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 检测结果
    all_results = {}
    
    # 检测低光照数据集
    if choice in ['1', '3']:
        lowlight_results_dir = RESULTS_DIR / f"lowlight_{timestamp}"
        lowlight_results = detect_dataset(
            model, LOWLIGHT_DIR, lowlight_results_dir, device, conf_thres
        )
        if lowlight_results:
            all_results['lowlight'] = {
                'dir': str(lowlight_results_dir),
                **lowlight_results
            }
    
    # 检测增强后数据集
    if choice in ['2', '3']:
        if not ENHANCED_DIR.exists():
            print(f"\n⚠️  增强后的数据集不存在: {ENHANCED_DIR}")
            print("请先运行: python enhance_with_enlightengan.py 增强图像")
        else:
            enhanced_results_dir = RESULTS_DIR / f"enhanced_{timestamp}"
            enhanced_results = detect_dataset(
                model, ENHANCED_DIR, enhanced_results_dir, device, conf_thres
            )
            if enhanced_results:
                all_results['enhanced'] = {
                    'dir': str(enhanced_results_dir),
                    **enhanced_results
                }
    
    # 保存总体结果
    if all_results:
        overall_results_path = RESULTS_DIR / f"detection_results_{timestamp}.json"
        with open(overall_results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "=" * 60)
        print(f"🎉 检测完成!")
        print(f"总体结果保存到: {overall_results_path}")
        
        # 如果同时检测了两个数据集，提示比较
        if 'lowlight' in all_results and 'enhanced' in all_results:
            print("\n下一步:")
            print("比较增强前后的检测效果:")
            print(f"python compare_detection_results.py {overall_results_path}")
    
    print("\n" + "=" * 60)
    print("检测过程结束!")

if __name__ == "__main__":
    main()