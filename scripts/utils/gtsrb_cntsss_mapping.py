#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GTSRB 43类到 CNTSSS 3类的映射工具
用于利用 CNTSSS 低光照数据集进行额外验证
"""

# GTSRB 43类交通标志
GTSRB_CLASSES = [
    'speed_20', 'speed_30', 'speed_50', 'speed_60', 'speed_70', 'speed_80',
    'speed_80_end', 'speed_100', 'speed_120',
    'no_overtaking', 'no_overtaking_trucks',
    'priority_at_next_intersection', 'priority_road', 'give_way', 'stop',
    'no_entry', 'no_entry_trucks', 'no_entry_one_way',
    'general_caution', 'dangerous_curve_left', 'dangerous_curve_right',
    'double_curve', 'bumpy_road', 'slippery_road', 'road_narrows_right',
    'road_works', 'traffic_signals', 'pedestrians', 'children_crossing',
    'bicycles_crossing', 'ice_or_snow', 'wild_animals_crossing',
    'end_of_all_speed_and_overtaking_limits',
    'turn_right_ahead', 'turn_left_ahead', 'ahead_only',
    'go_straight_or_right', 'go_straight_or_left',
    'keep_right', 'keep_left', 'roundabout_mandatory',
    'end_of_no_overtaking', 'end_of_no_overtaking_trucks'
]

# CNTSSS 3类分类
CNTSSS_CLASSES = ['prohibitory', 'mandatory', 'warning']

def create_mapping():
    """
    创建 GTSRB 43类 → CNTSSS 3类的映射字典
    """
    mapping = {}
    
    # 1. Prohibitory (禁止类) - 红色圆形，禁止标志
    prohibitory_keywords = [
        'no_entry', 'no_overtaking', 'speed_20', 'speed_30', 'speed_50',
        'speed_60', 'speed_70', 'speed_80', 'speed_100', 'speed_120',
        'no_entry_trucks', 'no_entry_one_way', 'give_way', 'stop'
    ]
    
    # 2. Mandatory (强制类) - 蓝色圆形，必须遵守
    mandatory_keywords = [
        'ahead_only', 'turn_right_ahead', 'turn_left_ahead',
        'go_straight_or_right', 'go_straight_or_left',
        'keep_right', 'keep_left', 'roundabout_mandatory',
        'priority_road', 'priority_at_next_intersection'
    ]
    
    # 3. Warning (警告类) - 黄色三角形，警告标志
    warning_keywords = [
        'general_caution', 'dangerous_curve_left', 'dangerous_curve_right',
        'double_curve', 'bumpy_road', 'slippery_road', 'road_narrows_right',
        'road_works', 'traffic_signals', 'pedestrians', 'children_crossing',
        'bicycles_crossing', 'ice_or_snow', 'wild_animals_crossing'
    ]
    
    # 映射
    for cls in GTSRB_CLASSES:
        cls_lower = cls.lower()
        
        # 检查是否匹配禁止类
        if any(keyword in cls_lower for keyword in prohibitory_keywords):
            mapping[cls] = 'prohibitory'
        # 检查是否匹配强制类
        elif any(keyword in cls_lower for keyword in mandatory_keywords):
            mapping[cls] = 'mandatory'
        # 默认为警告类
        else:
            mapping[cls] = 'warning'
    
    return mapping

def convert_labels_to_3class(label_file, output_file, mapping):
    """
    将GTSRB格式的标签文件转换为CNTSSS的3类格式
    
    Args:
        label_file: 原始标签文件路径
        output_file: 输出标签文件路径
        mapping: 类别映射字典
    """
    from pathlib import Path
    
    label_path = Path(label_file)
    output_path = Path(output_file)
    
    if not label_path.exists():
        return False
    
    # 创建反向映射：GTSRB类名 → 索引 → CNTSSS类名
    gtsrb_to_cntsss = {}
    for i, gtsrb_class in enumerate(GTSRB_CLASSES):
        gtsrb_to_cntsss[i] = mapping[gtsrb_class]
    
    # CNTSSS类别索引
    cntsss_indices = {name: idx for idx, name in enumerate(CNTSSS_CLASSES)}
    
    # 读取并转换标签
    new_lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            old_class_idx = int(parts[0])
            if old_class_idx in gtsrb_to_cntsss:
                new_class_name = gtsrb_to_cntsss[old_class_idx]
                new_class_idx = cntsss_indices[new_class_name]
                
                # 替换类别索引
                parts[0] = str(new_class_idx)
                new_lines.append(' '.join(parts) + '\n')
    
    # 保存转换后的标签
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    return True

def create_cntsss_yaml(output_path='configs/cntsss_dataset.yaml'):
    """
    创建CNTSSS数据集的YAML配置文件
    """
    import yaml
    
    config = {
        'path': 'data/cntsss',  # 需要用户修改
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 3,
        'names': {
            0: 'prohibitory',
            1: 'mandatory',
            2: 'warning'
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ CNTSSS YAML配置文件已创建: {output_path}")
    return output_path

def main():
    """主函数"""
    print("="*60)
    print("GTSRB ↔ CNTSSS 类别映射工具")
    print("="*60)
    
    # 创建映射
    mapping = create_mapping()
    
    print("\n📊 类别映射关系:")
    print(f"\n{'GTSRB类别':<40} → CNTSSS类别")
    print("-" * 60)
    
    for cntsss_class in CNTSSS_CLASSES:
        print(f"\n{cntsss_class.upper()}:")
        matching = [cls for cls, mapped in mapping.items() if mapped == cntsss_class]
        for cls in matching[:5]:  # 只显示前5个
            print(f"  • {cls}")
        if len(matching) > 5:
            print(f"  ... 共 {len(matching)} 个类别")
    
    # 创建YAML配置文件
    print("\n" + "="*60)
    create_cntsss_yaml()
    
    print("\n" + "="*60)
    print("💡 使用建议:")
    print("="*60)
    print("方案1: 使用CNTSSS进行跨数据集验证")
    print("  • 下载CNTSSS数据集")
    print("  • 使用 configs/cntsss_dataset.yaml")
    print("  • 单独训练一个3类模型")
    print("  • 对比你的方法与YOLO-LLTS")
    
    print("\n方案2: 将GTSRB转换为3类")
    print("  • 使用 convert_labels_to_3class() 函数")
    print("  • 将43类标签转换为3类")
    print("  • 在同一框架下对比")
    
    print("\n方案3: 保持43类，使用CNTSSS作为额外测试集")
    print("  • 下载CNTSSS数据集")
    print("  • 手动标注或使用现有标注")
    print("  • 进行泛化能力测试")
    
    return mapping

if __name__ == '__main__':
    from pathlib import Path
    mapping = main()


