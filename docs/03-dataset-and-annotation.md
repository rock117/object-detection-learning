# 03 · 数据集与标注

自定义训练前，数据格式必须正确。YOLO 检测常用 **图片 + 同名 txt 标签**。

---

## 1. 推荐目录结构

```text
my_dataset/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── val/
│       ├── 0100.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── 0001.txt
    │   └── ...
    └── val/
        ├── 0100.txt
        └── ...
```

规则：

- `images/train/0001.jpg` ↔ `labels/train/0001.txt`  
- 一张图可以有多行（多个物体）  
- 没有物体时可以是空 txt，或按工具约定处理

---

## 2. 标签文件格式

每一行：

```text
class_id x_center y_center width height
```

全部为 **相对图像宽高的归一化坐标**（0～1）。

示例：图宽 640、高 480；框左上 (160,120)、右下 (320,360)：

- 中心 x = (160+320)/2 / 640 = 0.375  
- 中心 y = (120+360)/2 / 480 = 0.5  
- 宽 = (320-160)/640 = 0.25  
- 高 = (360-120)/480 = 0.5  

若类别是 `person` 且 id=0：

```text
0 0.375 0.5 0.25 0.5
```

---

## 3. data.yaml 示例

```yaml
path: ../datasets/my_dataset   # 数据集根目录（相对或绝对）
train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
```

`nc`（类别数）可由 `names` 推断；有的配置会显式写 `nc: 3`。

项目里的 `configs/coco8.yaml` 是官方迷你集示例，用来先跑通训练。

---

## 4. 标注工具（任选）

| 工具 | 说明 |
|------|------|
| [Label Studio](https://labelstud.io/) | 功能全，可导出 YOLO |
| [CVAT](https://www.cvat.ai/) | 团队协作友好 |
| [Roboflow](https://roboflow.com/) | 标注 + 增强 + 导出一体化 |
| [LabelImg](https://github.com/HumanSignal/labelImg) | 轻量本地工具 |

入门建议：先标 **50～200 张**、**1～3 个类别**，验证集留 20% 左右。

---

## 5. 数据质量检查清单

- [ ] train/val 都有图和对应标签  
- [ ] 类别 id 从 0 连续编号，与 yaml 一致  
- [ ] 坐标都在 0～1  
- [ ] 没有框宽高为 0  
- [ ] 验证集类别覆盖与训练接近（别只出现在 train）  
- [ ] 难例（遮挡、小目标、光照差）有代表性样本  

快速抽查：用预训练或刚训几轮的模型在 val 上 `predict`，肉眼看框是否离谱。

---

## 6. 增强（概念）

Ultralytics 训练默认带 mosaic、翻转等增强。自己做数据集时：

- 先保证标注正确，再谈增强  
- 小数据可适当增强；增强过猛可能伤小目标或特殊外观  

下一篇：[04 · 学习路线图](./04-learning-roadmap.md)
