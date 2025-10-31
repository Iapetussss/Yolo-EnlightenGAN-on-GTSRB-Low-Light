"""
训练结果分析指南
帮助理解和分析 YOLOv8 训练结果
"""

import os
from pathlib import Path
import pandas as pd

print("=" * 80)
print("📊 YOLOv8 训练结果分析指南")
print("=" * 80)

# 查找最新的训练结果
results_dir = Path('runs/train')
if not results_dir.exists():
    print("\n❌ 没有找到训练结果目录")
    exit(1)

# 找到最新的训练文件夹
train_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
if not train_dirs:
    print("\n❌ 没有找到训练结果")
    exit(1)

latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
print(f"\n📁 分析目录: {latest_dir}")
print("=" * 80)

# 读取 results.csv
csv_file = latest_dir / 'results.csv'
if csv_file.exists():
    df = pd.read_csv(csv_file)
    df = df.iloc[:-1]  # 移除最后一行空行
    
    print("\n" + "=" * 80)
    print("📈 1. 训练过程概览")
    print("=" * 80)
    
    # 显示关键指标
    final_epoch = df.iloc[-1]
    first_epoch = df.iloc[0]
    
    print(f"\n总训练轮数: {len(df)}")
    print(f"训练时长: {final_epoch['time']/3600:.2f} 小时")
    
    print("\n" + "-" * 80)
    print("🎯 核心性能指标（最终）:")
    print("-" * 80)
    print(f"  mAP@0.5        : {final_epoch['metrics/mAP50(B)']*100:.2f}% ⭐")
    print(f"  mAP@0.5:0.95   : {final_epoch['metrics/mAP50-95(B)']*100:.2f}%")
    print(f"  Precision      : {final_epoch['metrics/precision(B)']*100:.2f}%")
    print(f"  Recall         : {final_epoch['metrics/recall(B)']*100:.2f}%")
    
    # 性能评价
    map50 = final_epoch['metrics/mAP50(B)']
    print("\n" + "-" * 80)
    print("📊 性能评价:")
    print("-" * 80)
    if map50 > 0.95:
        print("  🌟🌟🌟 优秀！模型性能非常好！")
    elif map50 > 0.85:
        print("  🌟🌟 很好！模型性能良好！")
    elif map50 > 0.70:
        print("  🌟 不错！模型基本可用，可以继续优化")
    else:
        print("  ⚠️  一般，建议继续训练或调整参数")
    
    # 改进幅度
    improvement = (final_epoch['metrics/mAP50(B)'] - first_epoch['metrics/mAP50(B)']) * 100
    print(f"\n  从第1轮到第{len(df)}轮，mAP@0.5 提升了: {improvement:.2f}%")
    
    print("\n" + "-" * 80)
    print("📉 损失值分析（越低越好）:")
    print("-" * 80)
    print(f"  Box Loss   : {final_epoch['val/box_loss']:.4f}")
    print(f"  Class Loss : {final_epoch['val/cls_loss']:.4f}")
    print(f"  DFL Loss   : {final_epoch['val/dfl_loss']:.4f}")
    
    # 过拟合检查
    print("\n" + "-" * 80)
    print("🔍 过拟合检查:")
    print("-" * 80)
    train_loss = final_epoch['train/box_loss']
    val_loss = final_epoch['val/box_loss']
    loss_gap = val_loss - train_loss
    
    if loss_gap < 0.05:
        print(f"  ✅ 良好 - 验证损失和训练损失接近")
    elif loss_gap < 0.15:
        print(f"  ⚠️  轻微过拟合 - 可接受范围")
    else:
        print(f"  ❌ 过拟合 - 建议增加数据增强或正则化")
    print(f"  训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
    
else:
    print("\n❌ 找不到 results.csv 文件")

# 分析可用的可视化文件
print("\n" + "=" * 80)
print("🖼️  2. 可视化文件说明")
print("=" * 80)

visualizations = {
    'results.png': '📊 训练曲线 - 显示损失、精度、召回率等随训练变化',
    'confusion_matrix.png': '🎯 混淆矩阵 - 显示哪些类别容易混淆',
    'confusion_matrix_normalized.png': '🎯 归一化混淆矩阵 - 百分比形式',
    'F1_curve.png': '📈 F1曲线 - 不同置信度阈值下的F1分数',
    'PR_curve.png': '📈 PR曲线 - Precision-Recall关系',
    'P_curve.png': '📈 精确率曲线',
    'R_curve.png': '📈 召回率曲线',
    'BoxF1_curve.png': '📈 Box F1曲线',
    'labels.jpg': '🏷️  训练数据标签分布',
    'labels_correlogram.jpg': '🏷️  标签相关性图',
    'train_batch0.jpg': '🖼️  训练批次示例（带标注）',
    'train_batch1.jpg': '🖼️  训练批次示例',
    'train_batch2.jpg': '🖼️  训练批次示例',
    'val_batch0_labels.jpg': '🖼️  验证批次真实标签',
    'val_batch0_pred.jpg': '🖼️  验证批次预测结果',
}

available_files = []
for filename, description in visualizations.items():
    filepath = latest_dir / filename
    if filepath.exists():
        available_files.append((filename, description))
        print(f"\n✅ {filename}")
        print(f"   {description}")

# 模型权重
print("\n" + "=" * 80)
print("💾 3. 模型权重文件")
print("=" * 80)

weights_dir = latest_dir / 'weights'
if weights_dir.exists():
    best_model = weights_dir / 'best.pt'
    last_model = weights_dir / 'last.pt'
    
    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024 * 1024)
        print(f"\n✅ best.pt - 最佳模型 ({size_mb:.2f} MB)")
        print(f"   路径: {best_model}")
        print(f"   📌 这是你应该使用的模型！")
    
    if last_model.exists():
        size_mb = last_model.stat().st_size / (1024 * 1024)
        print(f"\n✅ last.pt - 最后一轮模型 ({size_mb:.2f} MB)")
        print(f"   路径: {last_model}")

# 使用建议
print("\n" + "=" * 80)
print("💡 4. 如何查看和使用这些结果")
print("=" * 80)

print("""
1️⃣  查看训练曲线 (results.png):
   - 打开图片查看各项指标变化
   - 如果曲线平稳，说明已经收敛
   - 如果还在上升，可以继续训练

2️⃣  查看混淆矩阵 (confusion_matrix.png):
   - 对角线越亮，分类越准确
   - 非对角线的亮点表示容易混淆的类别
   - 可以针对性改进这些类别

3️⃣  查看预测示例 (val_batch*_pred.jpg):
   - 直观看到模型的检测效果
   - 绿色框 = 正确检测
   - 红色框 = 错误检测

4️⃣  测试自己的图片:
   python step8_test_single_image.py

5️⃣  在验证集上完整评估:
   python step7_evaluate_model.py
""")

print("=" * 80)
print(f"📂 完整结果目录: {latest_dir.absolute()}")
print("💡 用文件资源管理器打开查看所有图片")
print("=" * 80)

# 提供快捷打开命令
print("\n快速打开结果文件夹:")
print(f"explorer \"{latest_dir.absolute()}\"")

