# 🌙 CNTSSS 真实夜间场景实验指南

## 数据集信息

**CNTSSS (Chinese Night-Time Scene)**
- 真实夜间驾驶场景
- 训练集：3276 张图像
- 测试集：786 张图像
- 类别：3 类（交通标志、交通灯、车辆）

---

## 实验设计

### 目标
对比真实夜间场景下，图像增强对检测性能的影响

### 对比组

| 实验 | 数据 | 描述 |
|-----|------|------|
| 1. Baseline | 原始夜间图像 | 直接训练，无增强 |
| 2. 传统增强 | 增强后图像 | CLAHE + Gamma |
| 3. 温和增强 | 温和增强图像 | 轻度 CLAHE + Gamma |

---

## 快速开始

### 实验1：Baseline（夜间原图）

```powershell
python scripts/training/train_cntsss_baseline.py
```

**预计时间：** 2-3 小时（数据集较小）

---

### 实验2：传统增强

#### Step 1: 批量增强数据
```powershell
python batch_enhance_cntsss.py
```

选择增强方式：
- [1] 温和增强 ⭐ 推荐
- [2] 传统增强（Retinex + CLAHE）

#### Step 2: 训练模型
```powershell
python scripts/training/train_cntsss_enhanced.py
```

---

## 预期结果

### Baseline
- **无增强，真实夜间场景**
- 预期 mAP: 40-60%（夜间场景通常较低）

### 传统增强
- **增强后再训练**
- 预期 mAP: 50-70%（希望有提升）

### 对比
- **如果增强 > Baseline**：证明增强有效
- **如果增强 ≈ Baseline**：说明模型已能适应夜间
- **如果增强 < Baseline**：说明增强有害

---

## 与 GTSRB 实验对比

| 特性 | GTSRB | CNTSSS |
|-----|-------|--------|
| 场景 | 模拟低光照 | 真实夜间 |
| 数据量 | 26K+ | 3K+ |
| 类别 | 43 类 | 3 类 |
| 意义 | 方法验证 | 真实应用 |

**结合两个实验，结论更全面！**

---

## 注意事项

1. **数据量小**：训练快，但结果波动可能大
2. **类别少**：更容易收敛
3. **真实场景**：结果更有实际意义


