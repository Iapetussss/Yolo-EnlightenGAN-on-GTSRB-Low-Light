"""
步骤 3: 转换数据集格式
将 GTSRB 格式转换为 YOLO 格式
"""

import sys
from pathlib import Path

print("=" * 60)
print("🔄 步骤 3: 转换数据集格式")
print("=" * 60)

# 尝试读取上一步保存的路径
config_file = Path(__file__).parent / 'dataset_path.txt'

if config_file.exists():
    with open(config_file, 'r') as f:
        default_path = f.read().strip()
    print(f"\n检测到之前保存的数据集路径: {default_path}")
    print("如果要使用这个路径，直接按 Enter")
    print("如果要使用其他路径，请输入新路径")
    
    dataset_path = input("\nGTSRB 数据集路径 (按 Enter 使用默认): ").strip()
    
    if not dataset_path:
        dataset_path = default_path
else:
    print("\n请输入 GTSRB 数据集的完整路径:")
    print("例如: D:\\datasets\\GTSRB 或 D:/datasets/GTSRB")
    dataset_path = input("\nGTSRB 数据集路径: ").strip()

if not dataset_path:
    print("\n❌ 错误: 必须提供数据集路径")
    print("   请重新运行此脚本并输入正确的路径")
    sys.exit(1)

dataset_path = Path(dataset_path)

if not dataset_path.exists():
    print(f"\n❌ 错误: 路径不存在: {dataset_path}")
    print("   请检查路径是否正确")
    sys.exit(1)

# 设置输出路径
output_path = Path(__file__).parent.parent / 'traffic_sign_data' / 'original'

print(f"\n✅ 数据集路径: {dataset_path}")
print(f"✅ 输出路径: {output_path}")

# 确认
print("\n⚠️  注意:")
print("   - 转换过程可能需要 10-30 分钟")
print("   - 需要约 2-3 GB 的磁盘空间")
print("   - 会创建新的图片文件（PNG 格式）")

response = input("\n是否继续? (输入 yes 继续，其他键取消): ").strip().lower()

if response != 'yes':
    print("\n❌ 用户取消操作")
    sys.exit(0)

# 开始转换
print("\n" + "=" * 60)
print("开始转换数据集...")
print("=" * 60)

try:
    from data_preparation import GTSRBDatasetConverter
    
    # 创建转换器
    converter = GTSRBDatasetConverter(
        gtsrb_root=str(dataset_path),
        output_root=str(output_path)
    )
    
    # 执行转换
    converter.convert_all(val_ratio=0.2)
    
    print("\n" + "=" * 60)
    print("✅ 数据集转换完成！")
    print("=" * 60)
    
    # 保存输出路径
    output_config = Path(__file__).parent / 'converted_dataset_path.txt'
    with open(output_config, 'w') as f:
        f.write(str(output_path.absolute()))
    
    print(f"\n转换后的数据集位置: {output_path}")
    print("\n下一步: 运行 python step4_create_lowlight.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 转换过程中出现错误:")
    print("=" * 60)
    print(str(e))
    print("\n可能的原因:")
    print("1. 数据集结构不正确")
    print("2. 磁盘空间不足")
    print("3. 权限问题")
    print("\n请检查错误信息并修正后重试")
    sys.exit(1)

