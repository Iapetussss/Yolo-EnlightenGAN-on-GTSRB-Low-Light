#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估 CNTSSS EnlightenGAN 模型
使用增强后的测试集进行评估（保持训练测试分布一致）
"""

from pathlib import Path
from ultralytics import YOLO

def main():
    print("="*70)
    print("  CNTSSS EnlightenGAN 模型评估")
    print("="*70)
    
    # 路径配置
    model_path = Path("experiments/cntsss_enlightengan_mild_enhanced/run/weights/best.pt")
    data_yaml = Path("configs/cntsss_enlightengan_mild_dataset.yaml")
    
    # 检查文件
    if not model_path.exists():
        print(f"\n❌ 模型不存在: {model_path}")
        print("可能的位置：")
        # 尝试查找其他可能的路径
        possible_paths = list(Path("experiments").glob("cntsss_*_enhanced/run*/weights/best.pt"))
        if possible_paths:
            print("\n找到以下模型：")
            for i, p in enumerate(possible_paths, 1):
                print(f"  [{i}] {p}")
            choice = input(f"\n选择模型 [1-{len(possible_paths)}]: ").strip()
            try:
                model_path = possible_paths[int(choice) - 1]
            except:
                print("❌ 无效选择")
                return
        else:
            return
    
    if not data_yaml.exists():
        print(f"\n❌ 配置文件不存在: {data_yaml}")
        return
    
    print(f"\n📋 评估配置:")
    print(f"  模型: {model_path}")
    print(f"  配置: {data_yaml}")
    print(f"  测试数据: 增强后的测试集（保持分布一致）")
    
    print(f"\n开始评估...\n")
    print("="*70)
    
    try:
        # 加载模型
        model = YOLO(str(model_path))
        
        # 评估（使用增强后的测试集）
        results = model.val(
            data=str(data_yaml),
            split='test',
            batch=16,
            workers=4,
            device=0
        )
        
        # 获取指标
        map50 = results.results_dict['metrics/mAP50(B)']
        map50_95 = results.results_dict['metrics/mAP50-95(B)']
        precision = results.results_dict['metrics/precision(B)']
        recall = results.results_dict['metrics/recall(B)']
        
        # 打印结果
        print("\n" + "="*70)
        print("  ✅ 评估完成！")
        print("="*70)
        
        print(f"\n关键指标（增强测试集）:")
        print(f"  mAP@0.5:      {map50:.4f} ({map50*100:.1f}%)")
        print(f"  mAP@0.5:0.95: {map50_95:.4f} ({map50_95*100:.1f}%)")
        print(f"  Precision:    {precision:.4f} ({precision*100:.1f}%)")
        print(f"  Recall:       {recall:.4f} ({recall*100:.1f}%)")
        
        print(f"\n{'='*70}")
        print("  实验对比")
        print(f"{'='*70}\n")
        print(f"CNTSSS 实验结果：\n")
        print(f"1. Baseline (原始→原始):     mAP = 67.9%")
        print(f"2. EnlightenGAN (增强→增强): mAP = {map50*100:.1f}%")
        
        if map50 > 0.679:
            improvement = (map50 - 0.679) * 100
            print(f"\n✅ EnlightenGAN 提升了 {improvement:.1f}%！")
        elif map50 > 0.65:
            print(f"\n≈ EnlightenGAN 效果与 Baseline 接近")
        else:
            decline = (0.679 - map50) * 100
            print(f"\n⚠️ EnlightenGAN 下降了 {decline:.1f}%")
        
        print(f"\n{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

