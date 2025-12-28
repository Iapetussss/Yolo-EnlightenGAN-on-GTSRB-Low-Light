"""
诊断训练结果 - 检查 98.65% mAP 是否合理
"""

from pathlib import Path
import pandas as pd
import yaml

def check_dataset_split():
    """检查数据集划分是否正确"""
    print("\n" + "=" * 70)
    print("1. 检查数据集划分")
    print("=" * 70)
    
    # 读取 YAML 配置
    yaml_path = Path('traffic_signs_dataset.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    
    # 统计各集图像数
    splits = {}
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        if img_dir.exists():
            images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
            labels = list(label_dir.glob('*.txt'))
            splits[split] = {
                'images': len(images),
                'labels': len(labels)
            }
        else:
            splits[split] = {'images': 0, 'labels': 0}
    
    print(f"\n数据集路径: {dataset_path}")
    print("\n数据统计:")
    print(f"{'集合':<10} {'图像数':<10} {'标签数':<10} {'匹配':<10}")
    print("-" * 50)
    
    for split, counts in splits.items():
        match = "✅" if counts['images'] == counts['labels'] else "❌"
        print(f"{split:<10} {counts['images']:<10} {counts['labels']:<10} {match:<10}")
    
    # 检查是否有重叠
    print("\n检查数据泄露...")
    train_imgs = set([f.name for f in (dataset_path / 'images' / 'train').glob('*')])
    val_imgs = set([f.name for f in (dataset_path / 'images' / 'val').glob('*')])
    
    overlap = train_imgs & val_imgs
    if overlap:
        print(f"❌ 警告：训练集和验证集有 {len(overlap)} 个重复图像！")
        print(f"   这可能导致虚高的 mAP！")
        print(f"   示例：{list(overlap)[:5]}")
    else:
        print("✅ 训练集和验证集无重叠")
    
    return splits, len(overlap) > 0

def check_training_results():
    """检查训练结果"""
    print("\n" + "=" * 70)
    print("2. 分析训练结果")
    print("=" * 70)
    
    results_csv = Path('runs/train/gtsrb_enlightengan8/results.csv')
    
    if not results_csv.exists():
        print("❌ 未找到训练结果文件")
        return None
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # 最终结果
    final_row = df.iloc[-1]
    
    print("\n最终性能指标:")
    print(f"  mAP@0.5:      {final_row['metrics/mAP50(B)']:.4f} ({final_row['metrics/mAP50(B)']*100:.2f}%)")
    print(f"  mAP@0.5:0.95: {final_row['metrics/mAP50-95(B)']:.4f} ({final_row['metrics/mAP50-95(B)']*100:.2f}%)")
    print(f"  Precision:    {final_row['metrics/precision(B)']:.4f} ({final_row['metrics/precision(B)']*100:.2f}%)")
    print(f"  Recall:       {final_row['metrics/recall(B)']:.4f} ({final_row['metrics/recall(B)']*100:.2f}%)")
    
    # 检查是否过拟合
    print("\n训练 vs 验证损失:")
    try:
        train_box_loss = final_row['train/box_loss']
        val_box_loss = final_row['val/box_loss']
        
        print(f"  训练集 Box Loss: {train_box_loss:.4f}")
        print(f"  验证集 Box Loss: {val_box_loss:.4f}")
        
        if val_box_loss < train_box_loss * 0.8:
            print("  ⚠️  验证损失远低于训练损失，可能有问题！")
        elif val_box_loss > train_box_loss * 1.5:
            print("  ⚠️  过拟合：验证损失远高于训练损失")
        else:
            print("  ✅ 损失比例正常")
    except:
        print("  ⚠️  无法比较损失")
    
    # 检查收敛
    print("\n训练收敛分析:")
    last_5 = df.tail(5)
    map_variance = last_5['metrics/mAP50(B)'].std()
    print(f"  最后5轮 mAP 标准差: {map_variance:.6f}")
    
    if map_variance < 0.001:
        print("  ✅ 已充分收敛")
    elif map_variance < 0.01:
        print("  ✅ 基本收敛")
    else:
        print("  ⚠️  可能未完全收敛，可以继续训练")
    
    return df

def check_task_difficulty():
    """评估任务难度"""
    print("\n" + "=" * 70)
    print("3. 任务难度评估")
    print("=" * 70)
    
    print("\nGTSRB 交通标志检测的典型难度:")
    print("  • 类别数: 43 类")
    print("  • 图像质量: 高（真实拍摄）")
    print("  • 目标大小: 中等（交通标志通常占比较大）")
    print("  • 背景复杂度: 中等")
    print("  • 遮挡情况: 较少")
    
    print("\n领域基准（GTSRB 数据集）:")
    print("  • 正常光照 + YOLOv8n: 88-95% mAP@0.5")
    print("  • 低光照 + YOLOv8n（无增强）: 55-70% mAP@0.5")
    print("  • 低光照 + 传统增强: 75-88% mAP@0.5")
    print("  • 低光照 + 深度学习增强: 80-92% mAP@0.5")
    
    print("\n你的结果: 98.65% mAP@0.5")
    print("  → 远高于典型表现！")
    
    print("\n可能的原因:")
    print("  1. ✅ 数据增强效果特别好")
    print("  2. ✅ 训练策略优秀")
    print("  3. ⚠️  数据泄露（训练集和验证集重叠）")
    print("  4. ⚠️  验证集太简单")
    print("  5. ⚠️  评估方式有误")

def check_validation_cache():
    """检查验证缓存"""
    print("\n" + "=" * 70)
    print("4. 检查数据缓存")
    print("=" * 70)
    
    cache_files = [
        'yolo_dataset/images/train.cache',
        'yolo_dataset/images/val.cache',
        'yolo_dataset/labels/train.cache',
        'yolo_dataset/labels/val.cache',
    ]
    
    print("\n缓存文件:")
    for cache in cache_files:
        cache_path = Path(cache)
        if cache_path.exists():
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {cache} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {cache} (不存在)")
    
    print("\n💡 建议：删除缓存重新验证")
    print("   rm yolo_dataset/images/*.cache")
    print("   python step7_evaluate_model.py")

def detailed_class_analysis():
    """详细的类别分析"""
    print("\n" + "=" * 70)
    print("5. 类别级别分析")
    print("=" * 70)
    
    results_dir = Path('runs/train/gtsrb_enlightengan8')
    
    # 检查混淆矩阵
    confusion_matrix = results_dir / 'confusion_matrix.png'
    if confusion_matrix.exists():
        print(f"\n✅ 混淆矩阵: {confusion_matrix}")
        print("   打开查看是否对角线过于完美")
    
    # 检查每个类别的性能
    print("\n如果所有类别都接近 100%，可能有问题")
    print("正常情况下：")
    print("  • 简单类别（如 STOP）: 95-99%")
    print("  • 中等类别: 85-95%")
    print("  • 困难类别（小样本）: 70-85%")

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  训练结果诊断工具".center(70))
    print("  检查 98.65% mAP 是否合理".center(70))
    print("=" * 70)
    
    # 1. 检查数据集
    splits, has_overlap = check_dataset_split()
    
    # 2. 分析训练结果
    df = check_training_results()
    
    # 3. 任务难度
    check_task_difficulty()
    
    # 4. 缓存检查
    check_validation_cache()
    
    # 5. 类别分析
    detailed_class_analysis()
    
    # 总结
    print("\n" + "=" * 70)
    print("  诊断总结".center(70))
    print("=" * 70)
    
    issues = []
    
    if has_overlap:
        issues.append("❌ 严重：训练集和验证集有重叠")
    
    if df is not None:
        final_map = df.iloc[-1]['metrics/mAP50(B)']
        if final_map > 0.97:
            issues.append("⚠️  警告：mAP 异常高（>97%）")
    
    if not issues:
        print("\n✅ 未发现明显问题")
        print("\n可能的原因:")
        print("  1. GTSRB 本身是相对简单的数据集")
        print("  2. 图像增强效果确实很好")
        print("  3. YOLOv8n 在这个任务上表现优秀")
        print("\n建议:")
        print("  • 在测试集上评估（更准确）")
        print("  • 与其他论文对比")
        print("  • 可视化检查预测结果")
    else:
        print("\n⚠️  发现潜在问题:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n建议措施:")
        print("  1. 检查数据划分脚本")
        print("  2. 删除缓存重新评估")
        print("  3. 在真正的测试集上评估")
        print("  4. 可视化预测结果")
        print("  5. 查看混淆矩阵")
    
    print("\n" + "=" * 70)
    
    # 生成报告
    print("\n📄 下一步：")
    print("  1. 查看混淆矩阵:")
    print("     runs/train/gtsrb_enlightengan8/confusion_matrix.png")
    print("\n  2. 重新评估（删除缓存）:")
    print("     python step7_evaluate_model.py")
    print("\n  3. 测试单张图像:")
    print("     python step8_test_single_image.py")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 诊断出错: {e}")
        import traceback
        traceback.print_exc()

