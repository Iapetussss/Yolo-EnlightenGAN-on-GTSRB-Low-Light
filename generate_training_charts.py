import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建图表输出目录
output_dir = 'chart_outputs'
os.makedirs(output_dir, exist_ok=True)

def load_training_data():
    """加载训练数据"""
    csv_path = 'runs/detect/new_gtsrb_yolov8_v2/results.csv'
    if not os.path.exists(csv_path):
        print(f"❌ 找不到训练结果文件: {csv_path}")
        return None
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"✅ 成功加载训练数据，共 {len(df)} 个epoch")
    return df

def plot_loss_curves(df):
    """绘制损失曲线"""
    plt.figure(figsize=(14, 8))
    
    # 训练损失
    plt.subplot(2, 1, 1)
    plt.plot(df['epoch'], df['train/box_loss'], 'r-', label='box_loss')
    plt.plot(df['epoch'], df['train/cls_loss'], 'g-', label='cls_loss')
    plt.plot(df['epoch'], df['train/dfl_loss'], 'b-', label='dfl_loss')
    plt.title('训练损失随Epoch的变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 验证损失
    plt.subplot(2, 1, 2)
    plt.plot(df['epoch'], df['val/box_loss'], 'r--', label='box_loss')
    plt.plot(df['epoch'], df['val/cls_loss'], 'g--', label='cls_loss')
    plt.plot(df['epoch'], df['val/dfl_loss'], 'b--', label='dfl_loss')
    plt.title('验证损失随Epoch的变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 损失曲线已保存至: {output_path}")
    return output_path

def plot_metrics_curves(df):
    """绘制性能指标曲线"""
    plt.figure(figsize=(14, 10))
    
    # 精确率和召回率
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['metrics/precision(B)'], 'c-', label='精确率')
    plt.plot(df['epoch'], df['metrics/recall(B)'], 'm-', label='召回率')
    plt.title('精确率和召回率随Epoch的变化')
    plt.xlabel('Epoch')
    plt.ylabel('值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # mAP50和mAP50-95
    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], 'y-', label='mAP50')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'k-', label='mAP50-95')
    plt.title('mAP指标随Epoch的变化')
    plt.xlabel('Epoch')
    plt.ylabel('值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 学习率
    plt.subplot(2, 2, 3)
    plt.plot(df['epoch'], df['lr/pg0'], 'r-', label='学习率 pg0')
    plt.plot(df['epoch'], df['lr/pg1'], 'g-', label='学习率 pg1')
    plt.plot(df['epoch'], df['lr/pg2'], 'b-', label='学习率 pg2')
    plt.title('学习率随Epoch的变化')
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 训练时间
    plt.subplot(2, 2, 4)
    plt.plot(df['epoch'], df['time'], 'purple', marker='o')
    plt.title('每个Epoch的训练时间')
    plt.xlabel('Epoch')
    plt.ylabel('时间 (秒)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能指标曲线已保存至: {output_path}")
    return output_path

def plot_heatmap(df):
    """绘制相关性热力图"""
    # 选择相关的指标列
    metrics_cols = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
        'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
        'metrics/precision(B)', 'metrics/recall(B)',
        'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]
    
    # 计算相关性矩阵
    corr_matrix = df[metrics_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask,
                fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('训练指标相关性热力图')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 相关性热力图已保存至: {output_path}")
    return output_path

def plot_radar_chart(df):
    """绘制雷达图比较开始和结束的性能"""
    # 选择要比较的指标
    metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    metrics_labels = ['精确率', '召回率', 'mAP50', 'mAP50-95']
    
    # 初始值和最终值
    initial = df.iloc[0][metrics]
    final = df.iloc[-1][metrics]
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    # 准备数据
    initial = initial.tolist()
    final = final.tolist()
    initial += initial[:1]
    final += final[:1]
    metrics_labels += metrics_labels[:1]
    
    # 绘制雷达图
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # 绘制线条
    ax.plot(angles, initial, 'g-', linewidth=2, label='Epoch 1')
    ax.plot(angles, final, 'r-', linewidth=2, label=f'Epoch {len(df)}')
    
    # 填充区域
    ax.fill(angles, initial, 'g', alpha=0.25)
    ax.fill(angles, final, 'r', alpha=0.25)
    
    # 设置标签
    ax.set_thetagrids(np.degrees(angles)[:-1], metrics_labels[:-1])
    
    # 设置y轴范围
    ax.set_ylim(0, 1.1)
    
    # 添加标题和图例
    plt.title('训练开始和结束的性能指标比较', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 添加网格
    ax.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_radar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能雷达图已保存至: {output_path}")
    return output_path

def calculate_metrics_summary(df):
    """计算训练指标的摘要统计信息"""
    summary = {
        'total_epochs': len(df),
        'total_time': df['time'].sum() / 60,  # 转换为分钟
        'final_precision': df['metrics/precision(B)'].iloc[-1],
        'final_recall': df['metrics/recall(B)'].iloc[-1],
        'final_map50': df['metrics/mAP50(B)'].iloc[-1],
        'final_map50_95': df['metrics/mAP50-95(B)'].iloc[-1],
        'max_map50': df['metrics/mAP50(B)'].max(),
        'avg_epoch_time': df['time'].mean(),
        'first_epoch_loss': df[['train/box_loss', 'train/cls_loss', 'train/dfl_loss']].iloc[0].sum(),
        'last_epoch_loss': df[['train/box_loss', 'train/cls_loss', 'train/dfl_loss']].iloc[-1].sum()
    }
    return summary

def generate_charts():
    """生成所有图表"""
    print("🚀 开始生成YOLOv8训练可视化图表...")
    
    # 加载数据
    df = load_training_data()
    if df is None:
        return None
    
    # 生成图表
    charts = {
        'loss_curves': plot_loss_curves(df),
        'metrics_curves': plot_metrics_curves(df),
        'correlation_heatmap': plot_heatmap(df),
        'performance_radar': plot_radar_chart(df)
    }
    
    # 计算摘要统计
    summary = calculate_metrics_summary(df)
    
    print("🎉 图表生成完成！")
    print(f"📊 训练总轮数: {summary['total_epochs']}")
    print(f"⏱️  总训练时间: {summary['total_time']:.2f} 分钟")
    print(f"🎯 最终mAP50: {summary['final_map50']:.4f}")
    print(f"🎯 最终mAP50-95: {summary['final_map50_95']:.4f}")
    
    return charts, summary

if __name__ == "__main__":
    generate_charts()