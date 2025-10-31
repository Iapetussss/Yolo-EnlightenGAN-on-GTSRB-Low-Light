"""
检测报告生成脚本
用于生成低光照交通标志检测的详细报告
"""

import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np

print("=" * 60)
print("📝 生成交通标志检测报告")
print("=" * 60)

# 设置结果输出目录
REPORT_DIR = Path("results/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成交通标志检测报告")
    parser.add_argument("--results_file", help="检测结果JSON文件路径")
    parser.add_argument("--dataset_type", choices=["lowlight", "enhanced"], help="数据集类型")
    return parser.parse_args()

def load_detection_results(results_file=None, dataset_type=None):
    """加载检测结果"""
    if not results_file:
        # 自动查找最新的结果文件
        results_dir = Path("results")
        if not results_dir.exists():
            print(f"❌ 结果目录不存在: {results_dir}")
            print("请先运行: python detect_with_yolo.py")
            sys.exit(1)
        
        # 查找以detection_results_开头的JSON文件
        result_files = list(results_dir.glob("detection_results_*.json"))
        if not result_files:
            print("❌ 未找到检测结果文件")
            print("请先运行: python detect_with_yolo.py")
            sys.exit(1)
        
        # 选择最新的文件
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        print(f"\n✅ 自动加载最新结果文件: {latest_file}")
        results_file = latest_file
    
    # 检查文件是否存在
    if not Path(results_file).exists():
        print(f"❌ 结果文件不存在: {results_file}")
        sys.exit(1)
    
    # 加载结果
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 确定要分析的数据集
        available_datasets = []
        if 'lowlight' in results:
            available_datasets.append('lowlight')
        if 'enhanced' in results:
            available_datasets.append('enhanced')
        
        if not available_datasets:
            print("❌ 结果文件中没有有效的检测数据")
            sys.exit(1)
        
        # 如果没有指定数据集类型，让用户选择
        if not dataset_type:
            print("\n可用的检测结果:")
            for i, ds in enumerate(available_datasets, 1):
                print(f"{i}. {ds} 数据集")
            
            choice = input("请选择要生成报告的数据集 [1-{}]: ".format(len(available_datasets))).strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available_datasets):
                    dataset_type = available_datasets[idx]
                else:
                    dataset_type = available_datasets[0]
                    print(f"选择无效，使用默认: {dataset_type}")
            except:
                dataset_type = available_datasets[0]
                print(f"选择无效，使用默认: {dataset_type}")
        
        # 检查指定的数据集是否存在
        if dataset_type not in results:
            print(f"❌ 结果文件中没有 {dataset_type} 数据集的检测结果")
            sys.exit(1)
        
        return results[dataset_type], dataset_type
        
    except Exception as e:
        print(f"❌ 加载结果文件失败: {e}")
        sys.exit(1)

def calculate_detailed_metrics(detection_data):
    """计算详细的检测指标"""
    results = detection_data['results']
    stats = detection_data['stats']
    
    # 计算置信度分布
    confidences = []
    for result in results:
        for det in result['detections']:
            confidences.append(det['confidence'])
    
    # 计算检测数分布
    detection_counts = [result['num_detections'] for result in results]
    
    # 计算检测到目标的图像比例
    images_with_detections = sum([1 for r in results if r['num_detections'] > 0])
    detection_rate = (images_with_detections / stats['total_images']) * 100
    
    # 计算不同置信度区间的检测数量
    confidence_bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    confidence_distribution = {}
    for bin_start, bin_end in confidence_bins:
        count = sum(1 for conf in confidences if bin_start <= conf < bin_end)
        confidence_distribution[f"{bin_start:.2f}-{bin_end:.2f}"] = count
    
    # 计算平均置信度
    avg_confidence = np.mean(confidences) if confidences else 0
    median_confidence = np.median(confidences) if confidences else 0
    
    # 找出检测最多和最少的图像
    max_detections = max(detection_counts) if detection_counts else 0
    min_detections = min(detection_counts) if detection_counts else 0
    
    metrics = {
        'basic_stats': stats,
        'detection_rate': detection_rate,
        'avg_confidence': avg_confidence,
        'median_confidence': median_confidence,
        'confidence_distribution': confidence_distribution,
        'max_detections': max_detections,
        'min_detections': min_detections,
        'confidences': confidences,
        'detection_counts': detection_counts
    }
    
    return metrics

def generate_visualizations(metrics, output_dir, dataset_type):
    """生成可视化图表"""
    # 1. 检测数分布直方图
    plt.figure(figsize=(12, 6))
    
    # 检测数分布
    plt.subplot(1, 2, 1)
    plt.hist(metrics['detection_counts'], bins=range(max(metrics['detection_counts'])+2), 
             alpha=0.7, color='#4ECDC4', edgecolor='black')
    plt.title(f'每张图像检测数分布 ({dataset_type})')
    plt.xlabel('检测数量')
    plt.ylabel('图像数量')
    plt.grid(True, alpha=0.3)
    
    # 置信度分布
    plt.subplot(1, 2, 2)
    if metrics['confidences']:
        plt.hist(metrics['confidences'], bins=20, alpha=0.7, color='#FF6B6B', edgecolor='black')
        plt.axvline(metrics['avg_confidence'], color='red', linestyle='dashed', linewidth=2,
                   label=f'平均: {metrics["avg_confidence"]:.2f}')
        plt.legend()
    plt.title(f'检测置信度分布 ({dataset_type})')
    plt.xlabel('置信度')
    plt.ylabel('检测数量')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    hist_path = output_dir / f"detection_histograms_{dataset_type}.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 直方图已保存: {hist_path}")
    
    # 2. 置信度区间饼图
    if metrics['confidence_distribution']:
        plt.figure(figsize=(8, 6))
        labels = list(metrics['confidence_distribution'].keys())
        sizes = list(metrics['confidence_distribution'].values())
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title(f'置信度区间分布 ({dataset_type})')
        
        pie_path = output_dir / f"confidence_pie_{dataset_type}.png"
        plt.savefig(pie_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 饼图已保存: {pie_path}")
    
    return {
        'histogram': str(hist_path),
        'pie_chart': str(pie_path) if 'pie_path' in locals() else None
    }

def generate_report(detection_data, metrics, visualizations, dataset_type, output_dir):
    """生成检测报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"detection_report_{dataset_type}_{timestamp}.md"
    
    # 确定数据集类型的中文名称
    dataset_name = "低光照数据集" if dataset_type == "lowlight" else "增强后数据集"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 交通标志检测报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"数据集类型: **{dataset_name}**\n\n")
        
        f.write("## 1. 检测统计概览\n\n")
        
        f.write(f"- **总图像数量**: {metrics['basic_stats']['total_images']}\n")
        f.write(f"- **总检测数量**: {metrics['basic_stats']['total_detections']}\n")
        f.write(f"- **平均每张图像检测数**: {metrics['basic_stats']['avg_detections_per_image']:.2f}\n")
        f.write(f"- **检测到目标的图像比例**: {metrics['detection_rate']:.1f}%\n")
        f.write(f"- **平均检测置信度**: {metrics['avg_confidence']:.3f}\n")
        f.write(f"- **中位检测置信度**: {metrics['median_confidence']:.3f}\n")
        f.write(f"- **单张图像最大检测数**: {metrics['max_detections']}\n")
        f.write(f"- **单张图像最小检测数**: {metrics['min_detections']}\n\n")
        
        f.write("## 2. 置信度分布详情\n\n")
        f.write("| 置信度区间 | 检测数量 | 百分比 |\n")
        f.write("|------------|----------|--------|\n")
        
        total = sum(metrics['confidence_distribution'].values())
        for interval, count in metrics['confidence_distribution'].items():
            percentage = (count / total * 100) if total > 0 else 0
            f.write(f"| {interval} | {count} | {percentage:.1f}% |\n")
        f.write("\n")
        
        f.write("## 3. 可视化分析\n\n")
        f.write("### 3.1 检测数和置信度分布\n\n")
        if 'histogram' in visualizations:
            rel_path = Path(visualizations['histogram']).relative_to(REPORT_DIR)
            f.write(f"![检测数和置信度分布]({rel_path})\n\n")
        
        f.write("### 3.2 置信度区间比例\n\n")
        if 'pie_chart' in visualizations and visualizations['pie_chart']:
            rel_path = Path(visualizations['pie_chart']).relative_to(REPORT_DIR)
            f.write(f"![置信度区间分布]({rel_path})\n\n")
        
        f.write("## 4. 分析与结论\n\n")
        
        # 根据检测性能生成分析
        if metrics['detection_rate'] > 80:
            f.write("### 🔍 检测覆盖良好\n\n")
            f.write(f"模型成功检测到了{metrics['detection_rate']:.1f}%的图像中的交通标志，")
            f.write("检测覆盖范围很广，表明模型在当前数据集上表现良好。\n\n")
        elif metrics['detection_rate'] > 50:
            f.write("### 📊 检测覆盖中等\n\n")
            f.write(f"模型检测到了{metrics['detection_rate']:.1f}%的图像中的交通标志，")
            f.write("还有提升空间，建议调整检测参数或考虑模型优化。\n\n")
        else:
            f.write("### 📉 检测覆盖有限\n\n")
            f.write(f"模型仅检测到了{metrics['detection_rate']:.1f}%的图像中的交通标志，")
            f.write("检测效果不佳，建议：\n")
            f.write("1. 降低置信度阈值\n")
            f.write("2. 尝试使用更大的YOLO模型\n")
            f.write("3. 考虑数据增强或模型重新训练\n\n")
        
        # 置信度分析
        if metrics['avg_confidence'] > 0.7:
            f.write("### 🎯 高置信度检测\n\n")
            f.write(f"平均检测置信度达到{metrics['avg_confidence']:.3f}，")
            f.write("表明模型对大多数检测结果有很高的把握，误检率可能较低。\n\n")
        elif metrics['avg_confidence'] > 0.5:
            f.write("### ⚠️ 中等置信度检测\n\n")
            f.write(f"平均检测置信度为{metrics['avg_confidence']:.3f}，")
            f.write("检测结果可靠性一般，可能存在一定比例的误检。\n\n")
        else:
            f.write("### 🚨 低置信度检测\n\n")
            f.write(f"平均检测置信度仅为{metrics['avg_confidence']:.3f}，")
            f.write("检测结果可靠性较低，可能存在大量误检，建议提高模型性能。\n\n")
        
        # 针对数据集类型的特定建议
        if dataset_type == "lowlight":
            f.write("## 5. 低光照数据集特定建议\n\n")
            f.write("1. **考虑图像增强**: 低光照条件下检测难度较大，")
            f.write("建议使用图像增强技术提高图像质量。\n")
            f.write("2. **调整检测参数**: 对于低光照图像，")
            f.write("可能需要适当降低置信度阈值以捕获更多潜在目标。\n")
        else:
            f.write("## 5. 增强后数据集特定建议\n\n")
            f.write("1. **增强效果评估**: 增强后的图像应该能提供更好的检测性能，")
            f.write("建议与原始低光照图像的检测结果进行对比。\n")
            f.write("2. **参数优化**: 如果增强效果不明显，")
            f.write("可以尝试调整增强参数或更换增强方法。\n")
        
        f.write("## 6. 后续步骤建议\n\n")
        if dataset_type == "lowlight":
            f.write("1. 运行图像增强脚本: `python enhance_with_enlightengan.py`\n")
            f.write("2. 对增强后的图像进行检测: `python detect_with_yolo.py --choice 2`\n")
        else:
            f.write("1. 如果已检测低光照图像，运行对比分析: `python compare_detection_results.py`\n")
        f.write("2. 根据分析结果调整参数或模型\n")
        f.write("3. 尝试不同的置信度阈值或YOLO模型大小\n")
    
    print(f"✅ 检测报告已生成: {report_path}")
    return report_path

def display_summary(metrics, dataset_type):
    """显示检测结果摘要"""
    dataset_name = "低光照数据集" if dataset_type == "lowlight" else "增强后数据集"
    
    print("\n" + "=" * 60)
    print(f"📋 {dataset_name} 检测结果摘要")
    print("=" * 60)
    
    print(f"\n🔍 基本统计:")
    print(f"   总图像数: {metrics['basic_stats']['total_images']}")
    print(f"   总检测数: {metrics['basic_stats']['total_detections']}")
    print(f"   平均每张图像检测数: {metrics['basic_stats']['avg_detections_per_image']:.2f}")
    print(f"   检测到目标的图像比例: {metrics['detection_rate']:.1f}%")
    
    print(f"\n🎯 置信度指标:")
    print(f"   平均检测置信度: {metrics['avg_confidence']:.3f}")
    print(f"   中位检测置信度: {metrics['median_confidence']:.3f}")
    
    print(f"\n📊 检测分布:")
    print(f"   单张图像最大检测数: {metrics['max_detections']}")
    print(f"   单张图像最小检测数: {metrics['min_detections']}")
    
    print(f"\n📈 置信度区间分布:")
    for interval, count in metrics['confidence_distribution'].items():
        print(f"   {interval}: {count} 个检测")

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 加载检测结果
    detection_data, dataset_type = load_detection_results(args.results_file, args.dataset_type)
    
    # 计算详细指标
    metrics = calculate_detailed_metrics(detection_data)
    
    # 显示摘要
    display_summary(metrics, dataset_type)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPORT_DIR / f"{dataset_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成可视化
    visualizations = generate_visualizations(metrics, output_dir, dataset_type)
    
    # 生成报告
    report_path = generate_report(detection_data, metrics, visualizations, dataset_type, output_dir)
    
    print(f"\n" + "=" * 60)
    print(f"🎉 报告生成完成!")
    print(f"\n所有结果保存在: {output_dir}")
    print(f"详细报告: {report_path}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()