# 术语表（Glossary）

阅读标注实战、YOLO 多任务文档时遇到的词可在此查阅。按拼音/英文混排，**相关词互相引用**。

更完整的标注流程见：[06 · 真实项目里的标注实战](./06-annotation-in-practice.md)。

---

## A–C

### Active Learning（主动学习）
不随机标全量数据，而是让模型指出「自己最不确定」或「最可能标错」的样本，优先人工标注这些图，再用新标签重训，循环迭代。目的是用更少标注量达到接近全量标注的效果。

### Annotation / Labeling（标注）
给数据打「标准答案」的过程。检测标框、分割标 mask、分类标文件夹类别等。标注质量直接决定模型上限。

### Auto-label / Pre-label（自动标注 / 预标注）
先用已有模型（或 SAM 等工具）批量生成草稿标签，人再审核修改。真实项目的主流起点，而不是从零手动画。

### Backbone / Neck / Head
YOLO 网络三大块：骨干抽特征、颈部融合多尺度、检测头输出框/类/mask 等。见 [02 · YOLO 概览](./02-yolo-overview.md)。

### Bounding Box（边界框，BBox）
围住目标的矩形。水平框常用 `xyxy` 或 YOLO 归一化 `cx cy w h`；旋转框见 **OBB**。

### COCO
大规模检测/分割/姿态基准数据集与标注格式生态。YOLO 预训练常基于 COCO 或其子集（如 coco8）。

### Confidence（置信度，conf）
模型认为「该预测正确」的分数，常在 0～1。推理时低于 `--conf` 的框会被丢掉。

### CVAT
开源标注平台，支持检测框、多边形、关键点、视频跟踪传播等，团队协作常见。

---

## D–I

### Data Augmentation（数据增强）
训练时随机翻转、缩放、颜色抖动、Mosaic 等，扩充有效样本、减轻过拟合。不是「标注」，但常和数据策略一起谈。

### Depth Estimation（深度估计）
预测每个像素到相机的距离。真值多来自深度相机/双目/激光/仿真，几乎不靠人手「画深度」。见 [tasks/04-depth](./tasks/04-depth.md)。

### Domain Gap（域差异）
训练数据分布与真实上线场景不一致（光线、相机、国家、季节等），导致指标虚高、上线变差。缩小域差异靠采集真实场景数据、增强、微调。

### Fine-tune（微调）
在预训练权重上用自己的小数据继续训练，而不是随机初始化从头训。小项目默认应微调。

### Ground Truth（真值，GT）
标注给出的「正确答案」，评估时拿预测和 GT 比（如算 mAP、mIoU）。

### Human-in-the-loop（人机回环）
自动预标注 → 人改 → 再训练 → 再预标注的循环。真实标注产线的核心形态。

### Instance Segmentation（实例分割）
每个物体实例一块独立 mask（两个人 = 两块）。权重常带 `-seg`。对比 **Semantic Segmentation**。

### Inter-Annotator Agreement（标注者一致性）
多人标同一批数据时的一致程度。一致性差说明规范不清或任务过难，需要写指南、做质检。

### IoU（交并比）
两框（或两 mask）交集面积 / 并集面积。用于 NMS、匹配 GT、算检测/分割指标。见 [01 基础](./01-object-detection-basics.md)。

### ImageNet
大规模分类数据集。YOLO `-cls` 预训练常基于 ImageNet 类别。

---

## K–N

### Keypoint（关键点）
物体或人体上的有意义点（关节、角点等）。姿态估计输出一组关键点；标签里常带可见性 `v`。

### Label Studio
通用标注工具，可配置检测、分割、分类等，支持导出多种格式。

### mAP / mIoU
- **mAP**：检测/实例分割等常用，各类 AP 再平均。  
- **mIoU**：语义分割常用，各类 IoU 再平均。  

### Mask
像素级前景区域。实例分割是多块二值 mask；语义分割是一张类别 ID 图。

### Model Zoo
厂商提供的一批预训练权重（如 `yolo11n.pt`、`yolo11n-seg.pt`），开箱推理或当微调起点。

### NMS（非极大值抑制）
去掉与高分框 **IoU 过高** 的重复框，同一物体只留一个。参数常叫 `iou`。

### Negative Sample（负样本）
不含目标或应为「背景」的图。适量负样本可降低误报；全是正样本容易「到处乱检」。

---

## O–R

### OBB（Oriented Bounding Box）
带旋转角的框，适合航拍、细长倾斜目标。见 [tasks/07-obb](./tasks/07-obb.md)。

### OKS（Object Keypoint Similarity）
姿态评估里衡量关键点预测与真值接近程度的指标（类比框的 IoU）。

### Open Vocabulary / Grounding
不固定类别表、用文本描述找物体（如「红色的杯子」）。和闭集 COCO 80 类不同；本仓库入门以闭集 YOLO 为主。

### Pose Estimation（姿态估计）
检测人并估计骨架关键点。见 [tasks/06-pose](./tasks/06-pose.md)。

### Pretrain / Pretrained Weights（预训练权重）
在大数据集上先训好的模型参数。自己项目几乎总是从预训练开始微调。

### QA / QC（质检）
对标注结果抽检、仲裁、返工。没有质检的数据集，训练指标不可信。

### Review / Adjudication（审核 / 仲裁）
二级人员检查标注；两人冲突时第三人裁定。外包产线标配。

### Roboflow
数据与标注平台，预标注、增强、版本管理、导出 YOLO 格式较方便，个人/小团队常用。

---

## S

### SAM（Segment Anything Model）
Meta 提出的 **可提示分割** 基础模型：给你点、框或粗略提示，它生成物体 mask。  
**不替代**「这是什么类别」的分类本身，但能极大加速 **画轮廓**。常与检测框流水线结合：YOLO 出框 → SAM 出 mask → 人工微调。

典型用法：

1. 检测模型给出框  
2. 把框当作 prompt 喂给 SAM  
3. 得到实例 mask，导出为 YOLO-seg 或其它格式  

相关：Ultralytics 也集成了 SAM / SAM2 / MobileSAM / FastSAM 等，可做提示分割与辅助标注。

### SAM2（Segment Anything Model 2）
SAM 的后续版本，对 **视频** 更强：在一帧上提示后，可把 mask **传播到后续帧**，适合视频实例分割标注，大幅减少逐帧手绘。

和 SAM 对比（直觉）：

| | SAM | SAM2 |
|--|-----|------|
| 图片提示分割 | ✅ | ✅ |
| 视频跟踪式传播 | 弱 / 需外挂 | ✅ 原生更强 |
| 标注视频 | 可逐帧 | 更推荐 |

### Semantic Segmentation（语义分割）
每像素一个类别，**不区分**同类个体。YOLO26 权重常带 `-sem`。见 [tasks/03-semantic](./tasks/03-semantic-segment.md)。

### Semi-supervised / Weakly-supervised Learning
- **半监督**：少量精标 + 大量无标，用伪标签等技巧训练。  
- **弱监督**：只有弱标签（如只有图级标签、只有点、只有框）却想学分割等更强任务。  

降低标注成本的研究/工程方向，落地时要严格做验证集精标，防止伪标签漂移。

### Synthetic Data（合成数据）
用游戏引擎、3D、仿真渲染出图，并 **自动带精确标签**（框、mask、深度、关键点）。自动驾驶、机器人、仓储仿真很常见。注意与真实域的 Domain Gap。

### Seed Set（种子集）
第一批人工精标的小数据集（如 50～200 张），用来训练第一版模型，再开启预标注循环。

---

## T–Z

### Taxonomy / Label Spec（类别体系 / 标注规范）
写清楚：有哪些类、边界情况怎么判、遮挡/截断是否标、最小目标尺寸等。规范不清 → 标注者互掐 → 模型学乱。

### Teacher–Student / Pseudo-label（伪标签）
用较强模型（teacher）给无标数据打标签，再拿去训学生模型。必须过滤低置信、并保留干净验证集。

### Tracking（多目标跟踪）
跨帧给同一物体固定 ID（如 ByteTrack）。标注视频时也可借跟踪减少重复框选。检测/分割/姿态之上常接跟踪。

### Train / Val / Test Split
- **Train**：训练  
- **Val**：调参、早停、选最优权重  
- **Test**：最终报告（能不看就别看，防「刷测试集」）  

常见比例约 8:1:1 或 7:2:1；小数据至少保证 Val 有代表性。

### Transfer Learning（迁移学习）
把在 A 任务/数据上学到的表示用到 B。微调预训练 YOLO 就是迁移学习的一种。

### Ultralytics
YOLO 的主流工程实现之一，统一 `predict/train/val/export`，并支持多种任务与部分 SAM 生态。

### YOLO（You Only Look Once）
单阶段实时检测家族；现已扩展到分割、姿态、OBB、分类等。本仓库主线。

---

## 任务输出速查

| 英文任务名 | 中文 | 典型输出 |
|------------|------|----------|
| Detect | 目标检测 | boxes |
| Segment | 实例分割 | boxes + masks |
| Semantic | 语义分割 | semantic_mask |
| Depth | 深度估计 | depth map（米） |
| Classify | 图像分类 | probs |
| Pose | 姿态估计 | boxes + keypoints |
| OBB | 旋转框检测 | oriented boxes |

---

## 延伸阅读

- [06 · 真实项目里的标注实战](./06-annotation-in-practice.md)  
- [03 · 数据集与标注格式](./03-dataset-and-annotation.md)  
- [05 · 多任务总览](./05-yolo-tasks-overview.md)  
- SAM 论文与项目： [Segment Anything](https://segment-anything.com/)  
- Ultralytics SAM 文档： [SAM](https://docs.ultralytics.com/models/sam/) / [SAM2](https://docs.ultralytics.com/models/sam-2/)
