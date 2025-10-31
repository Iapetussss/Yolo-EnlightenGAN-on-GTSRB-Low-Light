"""
检测结果对比脚本
用于比较原始低光照图像和增强后图像的检测性能
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

print("=" * 60)
print("📊 比较低光照和增强后图像的检测效果")
print("=" * 60)

# 设置结果输出目录
COMPARE_DIR = Path("results/comparison")
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="比较低光照和增强后图像的检测效果")
    parser.add_argument("results_file", nargs="?", help="检测结果JSON文件路径")
    return parser.parse_args()

def load_detection_results(results_file=None):
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
        
        # 检查结果格式
        if 'lowlight' not in results or 'enhanced' not in results:
            print("❌ 结果文件格式不正确，缺少低光照或增强后的结果")
            print("请使用 --choice 3 运行 detect_with_yolo.py 来同时检测两个数据集")
            sys.exit(1)
        
        return results
        
    except Exception as e:
        print(f"❌ 加载结果文件失败: {e}")
        sys.exit(1)

def calculate_performance_metrics(lowlight_results, enhanced_results):
    """计算性能指标"""
    metrics = {}
    
    # 获取统计信息
    lowlight_stats = lowlight_results['stats']
    enhanced_stats = enhanced_results['stats']
    
    # 检测数量对比
    metrics['detection_count'] = {
        'lowlight': lowlight_stats['total_detections'],
        'enhanced': enhanced_stats['total_detections'],
        'improvement': enhanced_stats['total_detections'] - lowlight_stats['total_detections'],
        'improvement_percent': 0 if lowlight_stats['total_detections'] == 0 else \
            ((enhanced_stats['total_detections'] - lowlight_stats['total_detections']) / lowlight_stats['total_detections']) * 100
    }
    
    # 平均检测数对比
    metrics['avg_detections_per_image'] = {
        'lowlight': lowlight_stats['avg_detections_per_image'],
        'enhanced': enhanced_stats['avg_detections_per_image'],
        'improvement': enhanced_stats['avg_detections_per_image'] - lowlight_stats['avg_detections_per_image'],
        'improvement_percent': 0 if lowlight_stats['avg_detections_per_image'] == 0 else \
            ((enhanced_stats['avg_detections_per_image'] - lowlight_stats['avg_detections_per_image']) / lowlight_stats['avg_detections_per_image']) * 100
    }
    
    # 计算置信度分布
    lowlight_confidences = []
    enhanced_confidences = []
    
    for result in lowlight_results['results']:
        for det in result['detections']:
            lowlight_confidences.append(det['confidence'])
    
    for result in enhanced_results['results']:
        for det in result['detections']:
            enhanced_confidences.append(det['confidence'])
    
    metrics['confidence'] = {
        'lowlight_mean': np.mean(lowlight_confidences) if lowlight_confidences else 0,
        'enhanced_mean': np.mean(enhanced_confidences) if enhanced_confidences else 0,
        'lowlight_median': np.median(lowlight_confidences) if lowlight_confidences else 0,
        'enhanced_median': np.median(enhanced_confidences) if enhanced_confidences else 0,
        'confidence_improvement': np.mean(enhanced_confidences) - np.mean(lowlight_confidences) if lowlight_confidences and enhanced_confidences else 0
    }
    
    # 检测到目标的图像比例
    lowlight_images_with_detections = sum([1 for r in lowlight_results['results'] if r['num_detections'] > 0])
    enhanced_images_with_detections = sum([1 for r in enhanced_results['results'] if r['num_detections'] > 0])
    
    total_images = lowlight_stats['total_images']
    
    metrics['detection_rate'] = {
        'lowlight': (lowlight_images_with_detections / total_images) * 100,
        'enhanced': (enhanced_images_with_detections / total_images) * 100,
        'improvement': ((enhanced_images_with_detections - lowlight_images_with_detections) / total_images) * 100
    }
    
    return metrics

def visualize_comparison(metrics, output_dir):
    """可视化比较结果"""
    # 创建图表
    plt.figure(figsize=(15, 12))
    
    # 1. 总检测数对比
    plt.subplot(2, 2, 1)
    labels = ['低光照', '增强后']
    counts = [metrics['detection_count']['lowlight'], metrics['detection_count']['enhanced']]
    
    bars = plt.bar(labels, counts, color=['#FF6B6B', '#4ECDC4'])
    plt.title('总检测数量对比')
    plt.ylabel('检测数量')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 2. 平均检测数对比
    plt.subplot(2, 2, 2)
    avg_counts = [metrics['avg_detections_per_image']['lowlight'], metrics['avg_detections_per_image']['enhanced']]
    
    bars = plt.bar(labels, avg_counts, color=['#FF6B6B', '#4ECDC4'])
    plt.title('平均每张图像检测数量')
    plt.ylabel('平均检测数')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # 3. 平均置信度对比
    plt.subplot(2, 2, 3)
    confidences = [metrics['confidence']['lowlight_mean'], metrics['confidence']['enhanced_mean']]
    
    bars = plt.bar(labels, confidences, color=['#FF6B6B', '#4ECDC4'])
    plt.title('平均检测置信度')
    plt.ylabel('置信度')
    plt.ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 4. 检测率对比
    plt.subplot(2, 2, 4)
    detection_rates = [metrics['detection_rate']['lowlight'], metrics['detection_rate']['enhanced']]
    
    bars = plt.bar(labels, detection_rates, color=['#FF6B6B', '#4ECDC4'])
    plt.title('检测到目标的图像比例')
    plt.ylabel('百分比 (%)')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = output_dir / "performance_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 性能对比图表已保存: {chart_path}")
    
    # 显示图表
    plt.close()

def generate_comparison_report(metrics, output_dir):
    """生成比较报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comparison_report_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 低光照与增强后图像检测效果对比报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 性能指标对比\n\n")
        
        # 检测数量对比
        f.write("### 1.1 检测数量\n\n")
        f.write(f"- **低光照图像**: {metrics['detection_count']['lowlight']} 个检测\n")
        f.write(f"- **增强后图像**: {metrics['detection_count']['enhanced']} 个检测\n")
        f.write(f"- **改进**: +{metrics['detection_count']['improvement']} ({metrics['detection_count']['improvement_percent']:.1f}%)\n\n")
        
        # 平均检测数对比
        f.write("### 1.2 平均每张图像检测数\n\n")
        f.write(f"- **低光照图像**: {metrics['avg_detections_per_image']['lowlight']:.2f} 个/张\n")
        f.write(f"- **增强后图像**: {metrics['avg_detections_per_image']['enhanced']:.2f} 个/张\n")
        f.write(f"- **改进**: +{metrics['avg_detections_per_image']['improvement']:.2f} ({metrics['avg_detections_per_image']['improvement_percent']:.1f}%)\n\n")
        
        # 置信度对比
        f.write("### 1.3 检测置信度\n\n")
        f.write(f"- **低光照图像平均置信度**: {metrics['confidence']['lowlight_mean']:.3f}\n")
        f.write(f"- **增强后图像平均置信度**: {metrics['confidence']['enhanced_mean']:.3f}\n")
        f.write(f"- **改进**: +{metrics['confidence']['confidence_improvement']:.3f}\n\n")
        f.write(f"- **低光照图像中位置信度**: {metrics['confidence']['lowlight_median']:.3f}\n")
        f.write(f"- **增强后图像中位置信度**: {metrics['confidence']['enhanced_median']:.3f}\n\n")
        
        # 检测率对比
        f.write("### 1.4 检测率\n\n")
        f.write(f"- **低光照图像检测率**: {metrics['detection_rate']['lowlight']:.1f}%\n")
        f.write(f"- **增强后图像检测率**: {metrics['detection_rate']['enhanced']:.1f}%\n")
        f.write(f"- **改进**: +{metrics['detection_rate']['improvement']:.1f}%\n\n")
        
        # 分析总结
        f.write("## 2. 分析总结\n\n")
        
        if metrics['detection_count']['improvement_percent'] > 20:
            f.write("### 📈 显著改进\n\n")
            f.write("图像增强显著提高了检测性能，增强后的图像中能够检测到更多的交通标志。\n")
            f.write("这表明原始低光照图像中的交通标志由于光线不足而难以被检测算法识别。\n\n")
        elif metrics['detection_count']['improvement_percent'] > 0:
            f.write("### 📊 中度改进\n\n")
            f.write("图像增强对检测性能有一定的提升，增强后的图像中检测到的标志数量有所增加。\n\n")
        else:
            f.write("### 📉 无显著改进或下降\n\n")
            f.write("图像增强对检测数量没有明显提升，可能的原因：\n")
            f.write("1. 原始图像质量已经较好\n")
            f.write("2. 增强参数需要调整\n")
            f.write("3. 增强过程可能引入了噪声\n\n")
        
        if metrics['confidence']['confidence_improvement'] > 0.1:
            f.write("### 🎯 置信度大幅提升\n\n")
            f.write("增强后的图像不仅检测到更多目标，而且检测置信度也显著提高，\n")
            f.write("说明增强后的图像特征更加清晰，算法对检测结果更有把握。\n\n")
        
        if metrics['detection_rate']['improvement'] > 10:
            f.write("### ✅ 检测覆盖范围扩大\n\n")
            f.write("增强后能够在更多的图像中检测到交通标志，大大提高了系统的实用性。\n\n")
        
        # 建议
        f.write("## 3. 建议\n\n")
        f.write("### 3.1 进一步优化方向\n\n")
        f.write("1. **调整增强参数**: 根据实际效果调整CLAHE和Gamma校正参数\n")
        f.write("2. **尝试EnlightenGAN**: 如果尚未使用，建议下载EnlightenGAN模型进行对比\n")
        f.write("3. **调整YOLO参数**: 尝试不同的置信度阈值和NMS参数\n")
        f.write("4. **模型训练**: 使用增强后的数据集训练YOLO模型，可能会获得更好的效果\n\n")
        
        f.write("### 3.2 可视化分析\n\n")
        f.write("建议查看生成的性能对比图表，直观了解各项指标的改进情况。\n")
        f.write("同时可以检查检测结果中的可视化图像，分析具体哪些类型的标志检测效果提升明显。\n")
    
    print(f"✅ 比较报告已生成: {report_path}")
    return report_path

def display_summary(metrics):
    """显示比较结果摘要"""
    print("\n" + "=" * 60)
    print("📋 性能对比摘要")
    print("=" * 60)
    
    print(f"\n🔍 检测数量:")
    print(f"   低光照图像: {metrics['detection_count']['lowlight']} 个")
    print(f"   增强后图像: {metrics['detection_count']['enhanced']} 个")
    print(f"   改进: +{metrics['detection_count']['improvement']} ({metrics['detection_count']['improvement_percent']:.1f}%)")
    
    print(f"\n📊 平均每张图像检测数:")
    print(f"   低光照图像: {metrics['avg_detections_per_image']['lowlight']:.2f} 个/张")
    print(f"   增强后图像: {metrics['avg_detections_per_image']['enhanced']:.2f} 个/张")
    print(f"   改进: +{metrics['avg_detections_per_image']['improvement']:.2f} ({metrics['avg_detections_per_image']['improvement_percent']:.1f}%)")
    
    print(f"\n🎯 平均检测置信度:")
    print(f"   低光照图像: {metrics['confidence']['lowlight_mean']:.3f}")
    print(f"   增强后图像: {metrics['confidence']['enhanced_mean']:.3f}")
    print(f"   改进: +{metrics['confidence']['confidence_improvement']:.3f}")
    
    print(f"\n✅ 检测率 (检测到目标的图像比例):")
    print(f"   低光照图像: {metrics['detection_rate']['lowlight']:.1f}%")
    print(f"   增强后图像: {metrics['detection_rate']['enhanced']:.1f}%")
    print(f"   改进: +{metrics['detection_rate']['improvement']:.1f}%")

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 加载检测结果
    results = load_detection_results(args.results_file)
    
    # 计算性能指标
    metrics = calculate_performance_metrics(
        results['lowlight'], 
        results['enhanced']
    )
    
    # 显示摘要
    display_summary(metrics)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = COMPARE_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化结果
    visualize_comparison(metrics, output_dir)
    
    # 生成报告
    report_path = generate_comparison_report(metrics, output_dir)
    
    print(f"\n" + "=" * 60)
    print(f"🎉 比较分析完成!")
    print(f"\n所有结果保存在: {output_dir}")
    print(f"详细报告: {report_path}")
    print(f"性能对比图表: {output_dir / 'performance_comparison.png'}")
    print("\n结论:")
    
    # 生成简短结论
    if metrics['detection_count']['improvement_percent'] > 20:
        print("✅ 图像增强显著提高了交通标志检测性能!")
    elif metrics['detection_count']['improvement_percent'] > 0:
        print("✅ 图像增强对交通标志检测有积极影响。")
    else:
        print("⚠️  图像增强效果不明显，建议调整参数或尝试其他增强方法。")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()