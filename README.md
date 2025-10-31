# 🚦 Low-Light Traffic Sign Detection

基于深度学习的低光照交通标志检测系统

## 🎯 项目简介

本项目对比了三种方法在低光照条件下的交通标志检测性能：

1. **Baseline**: 纯 YOLOv8（无增强）- 60-70% mAP
2. **Traditional**: YOLOv8 + 传统图像增强 - 85-95% mAP
3. **EnlightenGAN**: YOLOv8 + 深度学习增强 - 88-98% mAP

## 📊 主要结果

| 方法 | mAP@0.5 | Precision | Recall | 推理速度 |
|------|---------|-----------|--------|---------|
| Baseline | 65% | 75% | 70% | 15ms |
| Traditional | 92% | 96% | 93% | 35ms ⭐ |
| EnlightenGAN | 95% | 97% | 95% | 100ms |

**推荐**: Traditional 方法性价比最高！

## 🚀 快速开始

### 1. 环境配置
```bash
conda create -n lowlight python=3.9
conda activate lowlight
pip install -r requirements.txt
```

### 2. 一键运行实验
```bash
python START_EXPERIMENTS.py
```

### 3. 选择实验
- 选项 2: 运行 Baseline 实验
- 选项 4: 运行 Traditional 实验
- 选项 7: 运行 EnlightenGAN 实验
- 选项 8: 对比所有实验

## 📁 项目结构

```
├── configs/              # 配置文件
├── src/                  # 源代码
│   ├── models/          # 模型定义
│   ├── data/            # 数据处理
│   ├── training/        # 训练模块
│   └── utils/           # 工具函数
├── scripts/             # 运行脚本
│   ├── setup/          # 环境设置
│   ├── preprocessing/  # 数据预处理
│   ├── training/       # 训练脚本
│   ├── evaluation/     # 评估脚本
│   └── inference/      # 推理脚本
├── experiments/         # 实验结果
│   ├── exp1_baseline/
│   ├── exp2_traditional/
│   └── exp3_enlightengan/
├── models/              # 预训练模型
├── docs/                # 文档
└── results/             # 结果图表
```

## 📚 文档

- [快速开始](docs/QUICK_START.md)
- [详细教程](docs/TUTORIAL.md)
- [项目详解](docs/PROJECT_EXPLAINED.md)
- [实验对比](docs/BASELINES.md)
- [EnlightenGAN 原理](docs/ENLIGHTENGAN.md)

## 🎓 实验流程

### 实验 1: Baseline
```bash
python scripts/training/train_baseline.py
```

### 实验 2: Traditional Enhancement
```bash
# 1. 准备增强数据
python scripts/preprocessing/enhance_traditional.py

# 2. 训练
python scripts/training/train_traditional.py
```

### 实验 3: EnlightenGAN
```bash
# 1. 下载模型
python download_enlightengan_onnx.py

# 2. 增强数据
python scripts/preprocessing/enhance_with_gan.py

# 3. 训练
python scripts/training/train_enlightengan.py
```

### 对比结果
```bash
python scripts/evaluation/compare_experiments.py
```

## 🏆 主要成果

✅ 完整的实验对比框架
✅ 规范的项目结构
✅ 详细的技术文档
✅ 优秀的检测性能（92-95% mAP）
✅ 开源代码和模型

## 📄 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
