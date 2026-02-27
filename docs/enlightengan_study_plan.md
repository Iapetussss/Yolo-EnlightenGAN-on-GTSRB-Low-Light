# 📚 EnlightenGAN 复现学习计划

## 🎯 目标
复现 EnlightenGAN 低光照增强模型，为PPT第1部分提供内容

---

## 📖 第一阶段：理论学习 (1-2天)

### 1.1 核心论文阅读

**主论文**：
- **标题**: EnlightenGAN: Deep Light Enhancement without Paired Supervision
- **会议**: IEEE TIP 2021
- **链接**: https://arxiv.org/abs/1906.06972
- **代码**: https://github.com/TAMU-VITA/EnlightenGAN

**阅读重点**：
- [ ] Abstract: 核心思想（无监督、自正则化）
- [ ] Introduction: 问题定义、现有方法局限
- [ ] Method 第3节:
  - [ ] 3.1 网络架构（生成器G、判别器D）
  - [ ] 3.2 自正则化感知损失（Self-Regularization）
  - [ ] 3.3 注意力机制（Attention U-Net）
- [ ] Experiments 第4节:
  - [ ] 数据集
  - [ ] 评估指标（PSNR、SSIM、NIQE）
  - [ ] 对比实验
- [ ] Results: 看图，理解效果

**做笔记**：
1. GAN的基本原理（生成对抗网络）
2. 为什么用无监督（paired data难获取）
3. 自正则化是什么（防止过度增强）
4. 如何训练（损失函数）

---

### 1.2 相关背景知识

**GAN基础** (必须懂):
- 生成器Generator：图像增强
- 判别器Discriminator：判断真假
- 对抗训练：G和D互相博弈

**推荐资源**：
- GAN原论文: Goodfellow 2014
- 视频教程: 李宏毅GAN系列（B站）
- 代码示例: PyTorch GAN Tutorial

**低光照增强方法对比**：
| 方法 | 类型 | 是否需要配对数据 | 效果 |
|-----|------|----------------|------|
| Retinex | 传统 | ❌ | 一般 |
| CLAHE | 传统 | ❌ | 一般 |
| Zero-DCE | 深度学习 | ❌ | 好 |
| EnlightenGAN | 深度学习 | ❌ | 很好 |
| LIME | 传统 | ❌ | 中等 |

---

### 1.3 整理PPT素材

**第1部分大纲**：

**Slide 1: 问题背景**
- 低光照图像的挑战
- 实际应用场景（夜间驾驶）
- 数据展示：原图 vs 增强图

**Slide 2: 技术路线**
- 传统方法局限性
- 深度学习方法
- EnlightenGAN的优势

**Slide 3: EnlightenGAN原理**
- GAN基础架构图
- 生成器G：Attention U-Net
- 判别器D：PatchGAN

**Slide 4: 自正则化机制**
- 为什么需要自正则化
- 公式解释
- 效果对比（有/无自正则化）

**Slide 5: 损失函数**
```
L_total = L_adv + λ1*L_per + λ2*L_self_reg
```
- 对抗损失：真实性
- 感知损失：内容保持
- 自正则化损失：防止过曝

---

## 🔧 第二阶段：代码复现 (2-3天)

### 2.1 环境准备

**创建新环境**（避免污染yoloen）:
```bash
conda create -n enlightengan python=3.8
conda activate enlightengan
```

**安装依赖**:
```bash
pip install torch torchvision
pip install opencv-python pillow
pip install tqdm matplotlib
pip install dominate visdom  # 可视化工具
```

---

### 2.2 下载官方代码

```bash
cd D:/rgznzuoye/
git clone https://github.com/TAMU-VITA/EnlightenGAN.git
cd EnlightenGAN
```

**代码结构**：
```
EnlightenGAN/
├── models/           # 模型定义
│   ├── enlighten_model.py
│   ├── networks.py
├── data/            # 数据加载
├── options/         # 训练参数
├── scripts/         # 训练脚本
├── test_dataset/    # 测试数据
└── train.py         # 主训练文件
```

---

### 2.3 准备数据

**方案A：使用官方测试数据**（快速验证）
```bash
# 官方提供了测试数据
cd test_dataset/
# 里面有一些低光照图像
```

**方案B：使用你的GTSRB数据**（与项目结合）
```bash
# 从你的低光照数据中选100张
python prepare_enlightengan_data.py
```

**数据组织**：
```
EnlightenGAN/
└── datasets/
    └── gtsrb_lowlight/
        ├── trainA/  # 低光照图像
        └── trainB/  # 正常光照图像（可选，无监督不需要）
```

---

### 2.4 下载预训练模型

**选项1：官方预训练模型**
```bash
# 从GitHub Release下载
cd checkpoints/
mkdir enlightening
cd enlightening
# 下载 latest_net_G.pth
```

**选项2：使用你已有的ONNX模型**
```bash
# 你已经有了: models/enlightengan/enlightengan.onnx
# 可以直接用
```

---

### 2.5 测试推理

**使用官方代码测试**：
```bash
python test.py \
  --dataroot ./test_dataset/ \
  --name enlightening \
  --model single \
  --no_dropout
```

**或使用你的ONNX模型**：
```python
# 你已经测试过了
python step_enlightengan_1_test.py
```

**输出**：
- 增强后的图像
- 对比图
- 量化指标

---

### 2.6 训练模型（可选）

**如果时间充裕，可以训练一个小模型**：

```bash
python train.py \
  --dataroot ./datasets/gtsrb_lowlight \
  --name my_enlightengan \
  --model single \
  --n_epochs 50 \
  --n_epochs_decay 50
```

**训练时间**：
- RTX 4060: 6-8小时（100 epochs）
- 可以只训练20 epochs看效果

---

## 📊 第三阶段：实验验证 (1-2天)

### 3.1 定量评估

**编写评估脚本**：
```python
# evaluate_enlightengan.py
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_enhancement(original, enhanced):
    # 亮度提升
    brightness_gain = np.mean(enhanced) - np.mean(original)
    
    # 对比度提升
    contrast_gain = np.std(enhanced) - np.std(original)
    
    # 熵（信息量）
    entropy_orig = calculate_entropy(original)
    entropy_enh = calculate_entropy(enhanced)
    
    return {
        'brightness_gain': brightness_gain,
        'contrast_gain': contrast_gain,
        'entropy_gain': entropy_enh - entropy_orig
    }
```

**在GTSRB数据上测试**：
- 选50张测试图
- 增强前后对比
- 计算指标

---

### 3.2 可视化对比

**生成对比图**：
```python
# 原图 | EnlightenGAN | 传统方法 | 正常光照
```

**制作PPT图表**：
- 柱状图：各指标对比
- 折线图：不同方法的性能
- 示例图：增强效果

---

### 3.3 整理复现文档

**撰写复现报告**：
```markdown
# EnlightenGAN 复现报告

## 1. 环境配置
- Python版本
- 依赖包

## 2. 复现步骤
- 下载代码
- 准备数据
- 运行推理

## 3. 实验结果
- 定量指标
- 可视化结果

## 4. 遇到的问题
- 问题1: 解决方法
- 问题2: 解决方法

## 5. 结论
- 复现是否成功
- 与论文对比
```

---

## 🎯 PPT第1部分内容整理

### Slide内容清单

**Slide 1: 低光照问题** ✅
- [ ] 3-4张夜间交通场景图
- [ ] 问题描述文字
- [ ] 研究意义

**Slide 2: 现有方法对比** ✅
- [ ] 传统方法（CLAHE、Retinex）示例
- [ ] 深度学习方法
- [ ] 对比表格

**Slide 3: EnlightenGAN架构** ✅
- [ ] GAN原理图
- [ ] 生成器结构图（从论文截取）
- [ ] 判别器结构

**Slide 4: 自正则化机制** ✅
- [ ] 公式
- [ ] 示意图（过曝vs正常）
- [ ] 效果对比

**Slide 5: 损失函数** ✅
- [ ] 三个损失项
- [ ] 权重设置
- [ ] 训练曲线

**Slide 6: 复现环境配置** ✅
- [ ] 硬件（RTX 4060）
- [ ] 软件（PyTorch、ONNX）
- [ ] 代码来源

**Slide 7: 推理测试** ✅
- [ ] 你已有的测试结果
- [ ] 3张对比图（原图、增强图、指标）

**Slide 8: 定量评估** ✅
- [ ] 亮度提升：+XX%
- [ ] 对比度提升：+XX%
- [ ] 处理速度：XX ms/图

**Slide 9: 与传统方法对比** ✅
- [ ] EnlightenGAN vs CLAHE
- [ ] 柱状图对比
- [ ] 优势分析

**Slide 10: 复现总结** ✅
- [ ] 成功复现 ✓
- [ ] 主要收获
- [ ] 为第3部分铺垫

---

## ⏰ 时间规划

| 任务 | 时间 | 完成标志 |
|-----|------|---------|
| 论文阅读 | 0.5天 | ✓ 理解原理 |
| 代码下载配置 | 0.5天 | ✓ 跑通推理 |
| 数据准备 | 0.5天 | ✓ 50张测试图 |
| 实验评估 | 1天 | ✓ 定量结果 |
| 可视化制作 | 0.5天 | ✓ 对比图表 |
| PPT整理 | 1天 | ✓ 10页slides |
| **总计** | **4天** | |

---

## 🎁 现成资源（你已有的）

✅ **已完成**：
1. EnlightenGAN ONNX模型已下载
2. 测试脚本已运行（`step_enlightengan_1_test.py`）
3. 有增强效果截图
4. 有GTSRB低光照数据

❌ **还需要**：
1. 理论理解（论文阅读）
2. 定量评估（计算指标）
3. 更多测试样例（50张）
4. 与传统方法的系统对比
5. PPT图表制作

---

## 💡 简化建议

**如果时间紧张**，可以这样：

### 精简版复现（2天完成）

**Day 1: 理论+快速测试**
- 上午：读论文Abstract+Method
- 下午：运行你的测试脚本，多测几张图

**Day 2: 评估+制作PPT**
- 上午：计算指标，做对比图
- 下午：整理成PPT（6-8页）

**不需要从头训练**，用预训练模型即可！

---

## 🚀 现在开始？

我帮你创建一个**快速启动脚本**：

