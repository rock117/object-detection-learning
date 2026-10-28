# 02 · YOLO 概览

YOLO = **You Only Look Once**：把检测当成一次回归问题，单次网络前向同时预测边界框和类别。

本项目使用 **Ultralytics** 实现（支持 YOLOv8 / YOLO11 等）。入门代码默认 **YOLO11n**。

---

## 1. 系列演进（极简）

| 代际 | 要点 |
|------|------|
| YOLOv1–v3 | 奠定「网格划分 + 直接回归」路线 |
| YOLOv4/v5 | 工程化、数据增强、训练技巧成熟 |
| YOLOv8 | Ultralytics 主流一代，任务统一（detect/seg/pose/cls） |
| YOLO11 | 更新的骨干与头结构，精度/速度再平衡 |

不必一开始深挖论文细节；先会用，再回头看结构图。

---

## 2. Ultralytics 里最常用的三件事

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")          # 加载预训练权重（自动下载）
results = model.predict("img.jpg")  # 推理
model.train(data="coco8.yaml", epochs=10)  # 训练
model.export(format="onnx")         # 导出
```

CLI 也行：

```bash
yolo detect predict model=yolo11n.pt source=bus.jpg
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10
yolo export model=yolo11n.pt format=onnx
```

---

## 3. 模型命名

`yolo11n.pt` 大致含义：

- `yolo11`：系列  
- `n`：规模（n/s/m/l/x）  
- `.pt`：PyTorch 权重  

任务后缀示例：

- `yolo11n.pt` —— 检测  
- `yolo11n-seg.pt` —— 实例分割  
- `yolo11n-pose.pt` —— 姿态  
- `yolo11n-cls.pt` —— 分类  
- `yolo11n-obb.pt` —— 旋转框  
- `yolo26n-sem.pt` —— 语义分割（YOLO26）  
- `yolo26n-depth.pt` —— 深度估计（YOLO26）  

多任务详解见 [05 · YOLO 多任务总览](./05-yolo-tasks-overview.md) 与 `docs/tasks/`。

---

## 4. 推理结果你该看什么

`results` 里常见字段（概念）：

- `boxes.xyxy`：像素坐标框  
- `boxes.conf`：置信度  
- `boxes.cls`：类别 id  
- `names`：id → 类别名 映射  

可视化：`result.plot()` 或设置 `save=True` 写到 `runs/detect/`。

---

## 5. 训练时关键参数

| 参数 | 含义 | 入门建议 |
|------|------|----------|
| `data` | 数据集 yaml | 先用 `coco8.yaml` |
| `epochs` | 训练轮数 | 10～50 试水 |
| `imgsz` | 输入边长 | 640 |
| `batch` | 批大小 | GPU 显存不够就改小，CPU 用 4/8 |
| `device` | `0` / `cpu` | 有卡用 GPU |
| `patience` | 早停耐心 | 可先默认 |

过拟合信号：训练集 loss 很好，验证集 mAP 不涨或下降 → 加数据、增强、早停、换更小模型。

---

## 6. 和「从零写 YOLO」的关系

本项目定位是 **用成熟库学检测流程**。若以后要深入：

1. 先会数据格式、训练曲线、错误案例分析  
2. 再读骨干 / Neck / Head、损失（box + cls + dfl 等）  
3. 最后才考虑改网络或复现论文  

下一篇：[03 · 数据集与标注](./03-dataset-and-annotation.md)
