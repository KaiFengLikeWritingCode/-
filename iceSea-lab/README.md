# 海冰智能识别实验指南（README）

> 版本：v1.0  
>  联系人：**KaiZhi Dong** (kaizhi@stu.ouc.edu.cn)
>  最近更新：2025-04-30

------

## 1 研究目的

基于 **Copernicus Marine Service (CMEMS)** 多年海洋物理重分析产品 `GLOBAL_MULTIYEAR_PHY_ENS_001_031`，
 实现：

1. **海冰厚度（`sithick_cglo`）+ 海表面高度异常（`zos_cglo`）→ 三分类掩码**
    *0 = 海水，1 = 海冰 (厚度 > 0)，2 = 陆地 (NaN)*
2. **语义分割模型**（U-Net / DeepLabV3+）训练与评估
3. 极区投影可视化、轮廓标注与批量推理结果输出

------

## 2 数据与许可

| 描述       | 详情                                                         |
| ---------- | ------------------------------------------------------------ |
| 产品       | GLOBAL_MULTIYEAR_PHY_ENS_001_031                             |
| 时段       | 2006-01-01 – 2010-01-01（示例文件 `glo_3ice.nc`）            |
| 变量       | `sithick_cglo` (海冰厚度 m)、`zos_cglo` (SSHA m)             |
| 空间分辨率 | 0.25° × 0.25°（示例：101 × 101 lat/lon）                     |
| 许可       | CMEMS Data License v4.4 — **务必在学术成果中注明数据来源与版权** |

------

## 3 环境依赖

```bash
# Python ≥ 3.9
pip install -r requirements.txt
requirements.txt
xarray
netCDF4
cartopy
matplotlib
albumentations
torch>=2.2
torchvision
segmentation-models-pytorch
tqdm
tensorboard
opencv-python>=4.6
numpy
cmocean          # 可选，若缺失自动退用 cividis
colorcet         # 可选，若缺失自动退用 cividis
```

> 若无法编译 Cartopy，可跳过极区地图绘制部分或使用 `hvplot.xarray` 替代。

------

## 4 项目结构

```
seaLab/
├── data/
│   ├── raw/               glo_3ice.nc
│   └── processed/
│       ├── train_ssha_png/   # 训练输入  (512×512 PNG)
│       ├── train_mask_png/   # 训练标签 lbl_*.png / onehot_*.npy
│       ├── test_ssha_png/    # 测试输入
│       └── test_mask_png/    # 测试标签
├── configs/
│   └── default.yaml          # 统一配置
├── src/
│   ├── make_dataset.py       # 数据集生成
│   ├── model.py              # U-Net / DeepLabV3+ 工厂
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 批量推理 + 轮廓叠加
│   └── utils.py              # IoU、色图等工具
└── results/                  # 日志、曲线、模型权重、可视化
```

------

## 5 数据预处理（`src/make_dataset.py`）

| 步骤            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| **灰度化 SSHA** | ① 判断分布对称性 ② 归一化到 [0,255] ③ Bicubic 插值至 512×512 |
| **掩码生成**    | 原网格判别：• `land_mask0 = NaN`• `ice_mask0 = sithick > 0`• 合成 `lbl0 ∈ {0,1,2}` |
| **放大**        | 整体最近邻放大至 512²，避免稀疏丢失                          |
| **划分**        | 最后一个自然年 (`last_year_start`) → 测试集，其余为训练集    |
| **输出**        | `ssha_*.png`、`lbl_*.png`、`onehot_*.npy`、调试 `mask_rgb_*.png` |

------

## 6 配置文件示例（`configs/default.yaml`）

```yaml
data:
  img_dir: data/processed/train_ssha_png
  lbl_dir: data/processed/train_mask_png

model:
  name: unet            # unet | deeplabv3+
  backbone: resnet34
  num_classes: 3
  pretrained_backbone: false

train:
  out_dir: results
  batch_size: 16
  epochs: 30
  lr: 1.0e-4
  weight_decay: 1.0e-4
  train_ratio: 0.8
  seed: 42
  device: cuda
  save_interval_epochs: 5

evaluation:
  weights: results/best_model.pth
  result_dir: results_overlay
  device: cuda
  colors: [[0,0,0],[0,128,255],[255,0,0]]   # BGR
  overlay_alpha: 0.45
```

------

## 7 训练与评估

### 7.1 数据集生成

```bash
python src/make_dataset.py
```

### 7.2 模型训练

```bash
python src/train.py --config configs/default.yaml
```

- **损失**：`CrossEntropyLoss`
- **优化器**：`AdamW`
- **日志**：TensorBoard (`results/tb_logs/`)
- **输出**：按 `save_interval_epochs` 保存检查点，自动记录 `best_model.pth`

### 7.3 批量推理 & 轮廓叠加

```bash
python src/evaluate.py \
       --config configs/default.yaml \
       --img_dir data/processed/test_ssha_png
```

输出 `overlay_*.png`：灰度底图 +
 蓝色海冰轮廓 / 红色陆地轮廓 + 标签文字。

------

## 8 复现步骤

```bash
# 1) 准备环境
conda create -n seaice python=3.10
conda activate seaice
pip install -r requirements.txt

# 2) 数据预处理
python src/make_dataset.py

# 3) 训练 U-Net
python src/train.py --config configs/default.yaml

# 4) 推理 + 可视化
python src/evaluate.py --config configs/default.yaml \
                       --img_dir data/processed/test_ssha_png
```

------

## 9 常见问题

| 问题                | 解决方案                                                     |
| ------------------- | ------------------------------------------------------------ |
| `lbl_*.png` 全黑    | 检查该日期海冰厚度最大值：`print(sithick.isel(time=i).max())`；若确为 0 则当天无冰 |
| 缺少 `icefire` 色带 | 已自动回退到 `cmocean.ice` / `cividis`；也可 `pip install colorcet` |
| Cartopy 安装失败    | 可跳过极区投影，可视化改用 `hvplot.xarray(geo=True)`         |

------

## 10 引用

如在论文或报告中使用本代码 / 数据，请至少引用：

1. **Copernicus Marine Service**

   > *GLOBAL_MULTIYEAR_PHY_ENS_001_031 — Ocean Physics Reanalysis Ensemble.*
   >  DOI: https://doi.org/10.xxxx/xxxxx

2. 本仓库（若已公开发布）：

   > **KaiZhi Dong** (2025) *Sea-Ice Semantic Segmentation Pipeline.*
   >  GitHub repository / Zenodo DOI.

