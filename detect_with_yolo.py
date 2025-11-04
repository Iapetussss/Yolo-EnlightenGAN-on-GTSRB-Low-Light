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
import argparse

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

def setup_yolo_model(model_path=None, device='0', conf_thres=0.25):
    """设置YOLOv8模型"""
    print("\n" + "=" * 60)
    print("加载YOLOv8模型...")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        # 如果没有提供模型路径或模型文件不存在，使用默认模型
        if model_path is None or not Path(model_path).exists():
            # 选择默认模型
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
                if not model_path or not Path(model_path).exists():
                    model_path = 'yolov8n.pt'
                    print("使用默认模型: yolov8n.pt")
            else:
                model_path = 'yolov8n.pt'
                print("使用默认模型: yolov8n.pt")
        
        print(f"\n加载模型: {model_path}")
        model = YOLO(model_path)
        
        print(f"使用设备: {device}")
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
    class_counts = {}
    confidence_values = []
    
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
                
                # 统计类别和置信度
                class_counts[cls] = class_counts.get(cls, 0) + 1
                confidence_values.append(conf)
                
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
    
    # 计算平均置信度
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
    print(f"- 平均检测置信度: {avg_confidence:.2f}")
    
    # 打印类别分布
    if class_counts:
        print(f"- 类别分布: {class_counts}")
    
    return {
        'results': detection_results,
        'stats': {
            'total_images': len(detection_results),
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections,
            'avg_confidence': avg_confidence,
            'class_counts': class_counts,
            'confidence_values': confidence_values
        }
    }

def generate_markdown_report(results, output_dir):
    """生成美观的markdown报告"""
    print(f"\n" + "=" * 60)
    print("生成检测报告...")
    print("=" * 60)
    
    # 创建报告目录
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成图表
    generate_charts(results['stats'], report_dir)
    
    # 创建markdown报告
    report_path = report_dir / "detection_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# YOLOv8 交通标志检测报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 检测统计部分
        f.write("## 检测统计\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 检测图像总数 | {results['stats']['total_images']} |\n")
        f.write(f"| 总检测目标数 | {results['stats']['total_detections']} |\n")
        f.write(f"| 平均每张图像检测数 | {results['stats']['avg_detections_per_image']:.2f} |\n")
        f.write(f"| 平均检测置信度 | {results['stats']['avg_confidence']:.2f} |\n\n")
        
        # 类别分布部分
        f.write("## 类别分布\n\n")
        f.write("### 饼图\n\n")
        f.write(f"![类别分布饼图](./charts/class_distribution_pie.png)\n\n")
        
        f.write("### 柱状图\n\n")
        f.write(f"![类别分布柱状图](./charts/class_distribution_bar.png)\n\n")
        
        # 置信度分布部分
        f.write("## 置信度分布\n\n")
        f.write(f"![置信度分布图](./charts/confidence_distribution.png)\n\n")
        
        # 检测示例部分
        f.write("## 检测示例\n\n")
        f.write("以下是部分检测结果示例（仅显示有检测目标的图像）:\n\n")
        
        # 添加一些检测示例图像
        test_dir = output_dir / "test"
        if test_dir.exists():
            example_images = list(test_dir.glob("det_*.jpg"))[:5]  # 最多5个示例
            for img_path in example_images:
                img_rel_path = f"../{img_path.relative_to(output_dir)}"
                f.write(f"### {img_path.name}\n\n")
                f.write(f"![检测结果]{img_rel_path} '检测结果'\n\n")
        
        # 结论部分
        f.write("## 结论\n\n")
        f.write("根据检测结果分析：\n\n")
        
        # 基于数据生成一些结论
        total_images = results['stats']['total_images']
        total_detections = results['stats']['total_detections']
        avg_confidence = results['stats']['avg_confidence']
        
        if total_detections == 0:
            f.write("- 在检测的图像中未发现任何交通标志，可能需要调整模型参数或重新训练模型。\n\n")
        else:
            f.write(f"- 模型成功检测到了{total_detections}个交通标志，平均每张图像检测{results['stats']['avg_detections_per_image']:.2f}个标志。\n\n")
            
            if avg_confidence > 0.7:
                f.write(f"- 平均检测置信度为{avg_confidence:.2f}，模型对检测结果有较高的可信度。\n\n")
            elif avg_confidence > 0.5:
                f.write(f"- 平均检测置信度为{avg_confidence:.2f}，模型检测结果可信度中等。\n\n")
            else:
                f.write(f"- 平均检测置信度为{avg_confidence:.2f}，模型检测结果可信度较低，建议调整置信度阈值或重新训练模型。\n\n")
        
        f.write("- 为了进一步提高检测性能，可以考虑：\n")
        f.write("  - 调整模型参数（如置信度阈值）\n")
        f.write("  - 扩充训练数据集，特别是低光照条件下的图像\n")
        f.write("  - 使用图像增强技术提高低光照图像的质量\n")
        f.write("  - 尝试不同的模型架构或训练策略\n")
    
    print(f"✅ 报告生成完成: {report_path}")
    return report_path

def generate_charts(stats, output_dir):
    """生成可视化图表"""
    # 创建图表目录
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 1. 类别分布饼图
    if stats['class_counts']:
        plt.figure(figsize=(10, 8))
        labels = [f"类别 {cls}" for cls in stats['class_counts'].keys()]
        sizes = list(stats['class_counts'].values())
        explode = [0.1] * len(labels)  # 突出显示所有部分
        
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # 保证饼图是圆的
        plt.title('交通标志类别分布')
        
        pie_chart_path = charts_dir / "class_distribution_pie.png"
        plt.savefig(str(pie_chart_path), bbox_inches='tight')
        plt.close()
    
    # 2. 类别分布柱状图
    if stats['class_counts']:
        plt.figure(figsize=(12, 6))
        classes = [f"类别 {cls}" for cls in stats['class_counts'].keys()]
        counts = list(stats['class_counts'].values())
        
        bars = plt.bar(classes, counts, color='skyblue')
        plt.xlabel('类别')
        plt.ylabel('检测数量')
        plt.title('各类别检测数量')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        bar_chart_path = charts_dir / "class_distribution_bar.png"
        plt.tight_layout()
        plt.savefig(str(bar_chart_path), bbox_inches='tight')
        plt.close()
    
    # 3. 置信度分布直方图
    if stats['confidence_values']:
        plt.figure(figsize=(10, 6))
        plt.hist(stats['confidence_values'], bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('置信度')
        plt.ylabel('频率')
        plt.title('检测置信度分布')
        plt.grid(True, alpha=0.3)
        
        # 添加均值线
        mean_conf = stats['avg_confidence']
        plt.axvline(mean_conf, color='red', linestyle='dashed', linewidth=2, label=f'均值: {mean_conf:.2f}')
        plt.legend()
        
        conf_chart_path = charts_dir / "confidence_distribution.png"
        plt.tight_layout()
        plt.savefig(str(conf_chart_path), bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='YOLOv8 交通标志检测')
    parser.add_argument('--model', type=str, default='runs/new_gtsrb_yolov8_v2/weights/best.pt',
                        help='YOLO模型路径')
    parser.add_argument('--device', type=str, default='0',
                        help='使用设备 (0/cpu)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='检测置信度阈值 (0-1)')
    parser.add_argument('--dataset', type=str, default='lowlight',
                        choices=['lowlight', 'enhanced', 'both'],
                        help='检测数据集')
    
    args = parser.parse_args()
    
    # 设置YOLO模型
    model, device, conf_thres = setup_yolo_model(args.model, args.device, args.conf_thres)
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 检测结果
    all_results = {}
    
    # 检测低光照数据集
    if args.dataset in ['lowlight', 'both']:
        lowlight_results_dir = RESULTS_DIR / f"lowlight_{timestamp}"
        lowlight_results = detect_dataset(
            model, LOWLIGHT_DIR, lowlight_results_dir, device, conf_thres
        )
        if lowlight_results:
            all_results['lowlight'] = {
                'dir': str(lowlight_results_dir),
                **lowlight_results
            }
            
            # 生成低光照数据集检测报告
            generate_markdown_report(lowlight_results, lowlight_results_dir)
    
    # 检测增强后数据集
    if args.dataset in ['enhanced', 'both']:
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
                
                # 生成增强后数据集检测报告
                generate_markdown_report(enhanced_results, enhanced_results_dir)
    
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