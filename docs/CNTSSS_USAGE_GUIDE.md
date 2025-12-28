# CNTSSS数据集使用指南

## 📚 关于CNTSSS数据集

**CNTSSS (Chinese Nighttime Traffic Sign Sample Set)** 是由 [YOLO-LLTS项目](https://github.com/linzy88/YOLO-LLTS) 提供的真实低光照交通标志数据集。

### 数据集特点

- ✅ **真实夜间数据**：在17个中国城市采集的真实夜间图像
- ✅ **多样化场景**：城市、高速公路、乡村环境
- ✅ **多种天气**：晴天、雨天
- ✅ **多种光照**：黄昏到深夜的各种光照条件
- ⚠️ **3类分类**：只有3个大类（prohibitory, mandatory, warning），不是43类细分类

---

## 🎯 与你的GTSRB项目的区别

| 项目 | GTSRB | CNTSSS |
|-----|-------|--------|
| **类别数** | 43类（细分类） | 3类（粗分类） |
| **数据类型** | 合成低光照（Gamma变换） | 真实夜间数据 |
| **数据来源** | 德国交通标志 | 中国17个城市 |
| **用途** | 细粒度分类 | 粗分类检测 |

---

## 💡 使用方案

### 方案1：跨数据集验证（推荐）⭐

**目的**：验证你的方法在真实低光照数据上的泛化能力

**步骤**：
1. 下载CNTSSS数据集
   ```bash
   # 从Google Drive或百度云下载
   # 链接：https://github.com/linzy88/YOLO-LLTS
   ```

2. 组织数据集结构
   ```
   data/
   └── cntsss/
       ├── train/
       │   ├── images/
       │   └── labels/
       └── test/
           ├── images/
           └── labels/
   ```

3. 修改配置文件路径
   ```yaml
   # configs/cntsss_dataset.yaml
   path: D:/rgznzuoye/new/data/cntsss
   ```

4. 训练一个3类模型
   ```bash
   python scripts/training/train_cntsss.py
   ```

5. 对比结果
   - 你的方法 vs YOLO-LLTS
   - 在真实低光照数据上的表现

**优势**：
- ✅ 验证泛化能力
- ✅ 真实数据更有说服力
- ✅ 可以作为PPT的额外亮点

---

### 方案2：将GTSRB转换为3类（实验扩展）

**目的**：在3类框架下对比，评估粗分类性能

**步骤**：
1. 运行映射脚本
   ```bash
   python scripts/utils/gtsrb_cntsss_mapping.py
   ```

2. 批量转换标签
   ```python
   # 使用 convert_labels_to_3class() 函数
   # 将43类标签转换为3类
   ```

3. 重新训练模型

**优势**：
- ✅ 与CNTSSS可比
- ✅ 评估粗分类性能

**劣势**：
- ❌ 丢失了细分类信息
- ❌ 与你的43类实验不兼容

---

### 方案3：保持独立实验（最简单）

**目的**：作为额外的验证实验

**步骤**：
1. 下载CNTSSS数据集
2. 使用 `configs/cntsss_dataset.yaml`
3. 单独训练一个3类模型
4. 在PPT中展示：
   - 实验1：GTSRB 43类（细分类）
   - 实验2：CNTSSS 3类（粗分类，真实低光照）

**优势**：
- ✅ 简单直接
- ✅ 不破坏现有实验
- ✅ 展示方法的多场景适用性

---

## 📊 实验设计建议

### 在你的PPT中可以这样展示：

**第4部分：后续任务与优化方向**

**Slide 1: 跨数据集验证**
- GTSRB（43类，合成低光照）
- CNTSSS（3类，真实低光照）
- 展示方法的泛化能力

**Slide 2: CNTSSS实验结果**
```
方法         mAP@0.5    mAP@0.5:0.95
Baseline      ?
传统增强      ?
EnlightenGAN  ?
YOLO-LLTS      ?
```

**Slide 3: 对比分析**
- 真实数据 vs 合成数据
- 粗分类 vs 细分类
- 不同场景下的表现

---

## 🚀 快速开始

### 1. 下载数据集

```bash
# 方式1: Google Drive
# 访问：https://drive.google.com/file/d/1A-7t-Wb5rjUZslUJ_1tltlUUvtSxBXdX/view

# 方式2: 百度云
# 链接：https://pan.baidu.com/s/1dEtWBVt6UWAKkaOYBq3uDg
# 提取码：dtrn
```

### 2. 解压并组织

```bash
# 解压后放到 data/cntsss/
data/
└── cntsss/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### 3. 修改配置文件

编辑 `configs/cntsss_dataset.yaml`，修改路径：
```yaml
path: D:/rgznzuoye/new/data/cntsss  # 你的实际路径
```

### 4. 运行映射工具（可选）

```bash
python scripts/utils/gtsrb_cntsss_mapping.py
```

会生成：
- 类别映射关系
- CNTSSS YAML配置文件

---

## 📝 注意事项

1. **类别不匹配**：CNTSSS只有3类，不能直接用于43类GTSRB实验
2. **数据格式**：CNTSSS已经是YOLO格式，可以直接使用
3. **标注格式**：确保标签文件格式正确（class_id x y w h）
4. **模型尺寸**：3类模型会更小，训练更快

---

## 💡 研究价值

使用CNTSSS可以：
- ✅ 展示方法的**真实场景适用性**
- ✅ 与**YOLO-LLTS**对比（已有SOTA结果）
- ✅ 验证**跨数据集泛化能力**
- ✅ 作为PPT的**额外亮点**

---

## 🔗 相关资源

- **项目主页**：https://github.com/linzy88/YOLO-LLTS
- **论文**：YOLO-LLTS: Real-Time Low-Light Traffic Sign Detection
- **数据集下载**：见项目README
- **预训练模型**：项目提供（3类模型）

---

## ❓ 常见问题

**Q: 能否将CNTSSS扩展到43类？**
A: 可以，但需要人工重新标注，工作量很大。

**Q: 3类模型能否用于43类任务？**
A: 不能，类别数不匹配，但可以用于粗分类任务。

**Q: 如何与我的43类实验对比？**
A: 作为独立实验，展示在不同场景下的表现。

---

## 🎯 推荐方案

**建议使用方案3（独立实验）**，因为：
1. ✅ 不破坏现有43类实验
2. ✅ 展示方法的多场景适用性
3. ✅ 作为PPT的额外亮点
4. ✅ 验证真实低光照场景下的表现

**时间成本**：
- 下载数据集：30分钟
- 组织数据：10分钟
- 训练3类模型：2-3小时
- 总计：约3-4小时

**研究成果**：
- 在真实低光照数据上的验证
- 与YOLO-LLTS的对比
- 跨数据集泛化能力验证


