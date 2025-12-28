# 🏗️ 项目重构方案

## 📋 目标

1. ✅ 规范项目结构（参考标准深度学习项目）
2. ✅ 清晰的实验对比框架
3. ✅ 专业的命名规范
4. ✅ 易于理解和维护

---

## 🎯 实验设计

### 对比实验

| 实验编号 | 名称 | 输入数据 | 增强方法 | 预期 mAP |
|---------|------|---------|---------|----------|
| **Exp 1** | Baseline | 低光照原图 | 无 | 60-70% |
| **Exp 2** | Traditional | 低光照原图 | CLAHE+Gamma+MSR | 85-95% |
| **Exp 3** | EnlightenGAN | 低光照原图 | EnlightenGAN | 88-98% |

**对比维度**：
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall
- 训练时间
- 推理速度
- 各类别性能

---

## 📁 新的项目结构

```
Low-Light-Traffic-Sign-Detection/
│
├── README.md                      # 项目说明
├── requirements.txt               # Python 依赖
├── LICENSE                        # 开源协议
├── .gitignore                     # Git 忽略文件
│
├── configs/                       # 配置文件目录
│   ├── dataset_config.yaml        # 数据集配置
│   ├── model_config.yaml          # 模型配置
│   └── train_config.yaml          # 训练配置
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始 GTSRB 数据
│   ├── processed/                 # 处理后的数据
│   │   ├── lowlight/              # 低光照数据
│   │   ├── enhanced_traditional/  # 传统方法增强
│   │   └── enhanced_enlightengan/ # EnlightenGAN 增强
│   └── yolo_format/               # YOLO 格式数据
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/                        # 模型相关
│   ├── yolov8/                    # YOLOv8 模型
│   │   └── yolov8n.pt
│   └── enlightengan/              # EnlightenGAN 模型
│       └── enlightengan.onnx
│
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── data/                      # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataset_loader.py      # 数据加载
│   │   ├── preprocess.py          # 数据预处理
│   │   └── augmentation.py        # 数据增强
│   │
│   ├── models/                    # 模型模块
│   │   ├── __init__.py
│   │   ├── detector.py            # 检测器基类
│   │   ├── yolo_detector.py       # YOLO 检测器
│   │   └── enlightengan.py        # EnlightenGAN 增强
│   │
│   ├── training/                  # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py             # 训练器
│   │   └── evaluator.py           # 评估器
│   │
│   └── utils/                     # 工具函数
│       ├── __init__.py
│       ├── logger.py              # 日志工具
│       ├── visualization.py       # 可视化工具
│       └── metrics.py             # 评估指标
│
├── experiments/                   # 实验目录
│   ├── exp1_baseline/             # 实验1：基线
│   │   ├── config.yaml
│   │   ├── train.log
│   │   └── results/
│   │
│   ├── exp2_traditional/          # 实验2：传统增强
│   │   ├── config.yaml
│   │   ├── train.log
│   │   └── results/
│   │
│   └── exp3_enlightengan/         # 实验3：EnlightenGAN
│       ├── config.yaml
│       ├── train.log
│       └── results/
│
├── scripts/                       # 运行脚本
│   ├── setup/                     # 环境设置脚本
│   │   ├── check_environment.py
│   │   ├── download_data.py
│   │   └── prepare_dataset.py
│   │
│   ├── preprocessing/             # 预处理脚本
│   │   ├── create_lowlight.py
│   │   ├── enhance_traditional.py
│   │   └── enhance_enlightengan.py
│   │
│   ├── training/                  # 训练脚本
│   │   ├── train_baseline.py
│   │   ├── train_traditional.py
│   │   └── train_enlightengan.py
│   │
│   ├── evaluation/                # 评估脚本
│   │   ├── evaluate_model.py
│   │   └── compare_experiments.py
│   │
│   └── inference/                 # 推理脚本
│       ├── predict_single.py
│       └── predict_batch.py
│
├── notebooks/                     # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_visualization.ipynb
│   └── 03_results_analysis.ipynb
│
├── docs/                          # 文档目录
│   ├── INSTALLATION.md            # 安装指南
│   ├── QUICK_START.md             # 快速开始
│   ├── EXPERIMENTS.md             # 实验说明
│   ├── API_REFERENCE.md           # API 文档
│   └── RESULTS.md                 # 结果报告
│
├── results/                       # 结果目录
│   ├── figures/                   # 图表
│   ├── tables/                    # 表格数据
│   └── comparison/                # 对比结果
│
└── tests/                         # 测试代码
    ├── test_data.py
    ├── test_models.py
    └── test_training.py
```

---

## 🔄 从旧结构迁移

### 旧文件 → 新位置

| 旧文件 | 新位置 | 说明 |
|--------|--------|------|
| `step1_check_environment.py` | `scripts/setup/check_environment.py` | 环境检查 |
| `step2_auto_download_dataset.py` | `scripts/setup/download_data.py` | 数据下载 |
| `step3_convert_dataset_kaggle.py` | `scripts/setup/prepare_dataset.py` | 数据准备 |
| `step4_create_lowlight.py` | `scripts/preprocessing/create_lowlight.py` | 低光照生成 |
| `step5_enhance_images.py` | `scripts/preprocessing/enhance_traditional.py` | 传统增强 |
| `step5_enhance_images_enlightengan.py` | `scripts/preprocessing/enhance_enlightengan.py` | GAN增强 |
| `step6_train_model.py` | `scripts/training/train_traditional.py` | 训练（传统） |
| `step7_evaluate_model.py` | `scripts/evaluation/evaluate_model.py` | 评估 |
| `step8_test_single_image.py` | `scripts/inference/predict_single.py` | 单图预测 |
| `enlightened_gtsrb.py` | `src/models/detector.py` | 核心检测器 |
| `enlightengan_inference.py` | `src/models/enlightengan.py` | GAN推理 |
| `reorganize_dataset.py` | `scripts/setup/reorganize_data.py` | 数据重组 |

---

## 📝 命名规范

### 文件命名
- **脚本**：`动词_名词.py`（如 `train_model.py`）
- **模块**：`名词.py`（如 `detector.py`）
- **配置**：`名词_config.yaml`（如 `dataset_config.yaml`）

### 变量命名
- **类**：`PascalCase`（如 `YOLODetector`）
- **函数**：`snake_case`（如 `train_model`）
- **常量**：`UPPER_SNAKE_CASE`（如 `MAX_EPOCHS`）

### 实验命名
```
exp{编号}_{简短描述}/
例如：
exp1_baseline/
exp2_traditional/
exp3_enlightengan/
```

---

## 🚀 实施步骤

### 阶段 1：创建新结构（15分钟）
```bash
python scripts/restructure/create_new_structure.py
```

### 阶段 2：迁移现有文件（20分钟）
```bash
python scripts/restructure/migrate_files.py
```

### 阶段 3：更新导入路径（10分钟）
```bash
python scripts/restructure/update_imports.py
```

### 阶段 4：验证功能（10分钟）
```bash
python scripts/restructure/verify_migration.py
```

---

## 🎯 实验执行计划

### 实验 1：Baseline（纯 YOLOv8，无增强）

**目标**：建立性能基线

```bash
# 1. 准备数据（低光照，无增强）
python scripts/preprocessing/create_lowlight.py

# 2. 训练
python scripts/training/train_baseline.py \
    --data configs/exp1_baseline.yaml \
    --epochs 20 \
    --batch 2 \
    --name exp1_baseline

# 3. 评估
python scripts/evaluation/evaluate_model.py \
    --model experiments/exp1_baseline/weights/best.pt \
    --data configs/exp1_baseline.yaml
```

**预期结果**：60-70% mAP@0.5

---

### 实验 2：Traditional Enhancement

**目标**：验证传统增强方法

```bash
# 1. 图像增强
python scripts/preprocessing/enhance_traditional.py

# 2. 训练
python scripts/training/train_traditional.py \
    --data configs/exp2_traditional.yaml \
    --epochs 20 \
    --batch 2 \
    --name exp2_traditional

# 3. 评估
python scripts/evaluation/evaluate_model.py \
    --model experiments/exp2_traditional/weights/best.pt \
    --data configs/exp2_traditional.yaml
```

**预期结果**：85-95% mAP@0.5

---

### 实验 3：EnlightenGAN Enhancement

**目标**：验证深度学习增强方法

```bash
# 1. 下载 EnlightenGAN 模型
python scripts/preprocessing/download_enlightengan.py

# 2. 图像增强
python scripts/preprocessing/enhance_enlightengan.py

# 3. 训练
python scripts/training/train_enlightengan.py \
    --data configs/exp3_enlightengan.yaml \
    --epochs 20 \
    --batch 2 \
    --name exp3_enlightengan

# 4. 评估
python scripts/evaluation/evaluate_model.py \
    --model experiments/exp3_enlightengan/weights/best.pt \
    --data configs/exp3_enlightengan.yaml
```

**预期结果**：88-98% mAP@0.5

---

### 实验对比

```bash
# 生成对比报告
python scripts/evaluation/compare_experiments.py \
    --exp1 experiments/exp1_baseline \
    --exp2 experiments/exp2_traditional \
    --exp3 experiments/exp3_enlightengan \
    --output results/comparison/
```

**输出**：
- `comparison_table.csv`：性能对比表
- `comparison_curves.png`：训练曲线对比
- `comparison_report.md`：详细报告

---

## 📊 预期对比结果

| 指标 | Exp1 (Baseline) | Exp2 (Traditional) | Exp3 (EnlightenGAN) |
|------|----------------|-------------------|-------------------|
| **mAP@0.5** | 60-70% | 85-95% | 88-98% |
| **mAP@0.5:0.95** | 45-55% | 75-85% | 78-92% |
| **Precision** | 70-80% | 90-96% | 92-98% |
| **Recall** | 65-75% | 88-94% | 90-97% |
| **训练时间** | ~11h | ~12h | ~12h |
| **推理速度** | 15ms | 35ms | 100ms |

**结论**：
- Baseline → Traditional: **+20-25% mAP**
- Traditional → EnlightenGAN: **+3-5% mAP**
- 传统方法**性价比最高**（速度快，效果好）

---

## 🎓 答辩时的展示

### PPT 结构建议

**Slide 1-3: 引入**
- 问题：低光照下交通标志检测困难
- 动机：提升检测性能
- 方法：对比三种方案

**Slide 4-6: 方法**
- Exp1: Baseline（YOLOv8）
- Exp2: Traditional Enhancement
- Exp3: EnlightenGAN Enhancement

**Slide 7-12: 实验**
- 数据集：GTSRB
- 实验设置：统一参数
- 三个实验的详细配置

**Slide 13-18: 结果**
- 对比表格
- 训练曲线
- 混淆矩阵
- 可视化示例

**Slide 19-20: 分析**
- 为什么传统方法效果好？
- EnlightenGAN 的优劣？
- 实用性建议

**Slide 21-22: 总结**
- 主要贡献
- 局限性
- 未来工作

---

## ✅ 优势

重构后的项目具有：

1. **专业性** ⭐⭐⭐⭐⭐
   - 符合深度学习项目标准
   - 清晰的模块划分
   - 规范的命名

2. **可维护性** ⭐⭐⭐⭐⭐
   - 代码组织清晰
   - 易于添加新实验
   - 便于他人理解

3. **可复现性** ⭐⭐⭐⭐⭐
   - 配置文件管理
   - 详细的实验记录
   - 脚本化流程

4. **可展示性** ⭐⭐⭐⭐⭐
   - 结构一目了然
   - 实验对比清晰
   - 文档完善

---

## 📞 下一步

1. **我会创建自动重构脚本**
2. **保留你的原始数据和模型**
3. **创建新的规范结构**
4. **逐步迁移并测试**

准备好了吗？让我开始重构！🚀

