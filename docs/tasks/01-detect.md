# 任务 01 · 目标检测（Detect）

## 1. 它是什么

对图像中的每个目标预测：

- **位置**：水平矩形框（通常 `xyxy`）
- **类别**：class id + 名称
- **置信度**：模型认为「这是该类」的分数

不输出像素级轮廓，也不输出关节点。

## 2. 适用场景

| 适合 | 不太适合 |
|------|----------|
| 安防入侵、车流统计、货架商品检测 | 需要精确轮廓/面积 |
| 实时摄像头报警 | 整图只有一个主题（用分类更简单） |
| 作为跟踪、抠图流水线的前端 | 目标严重倾斜且框浪费大（考虑 OBB） |

## 3. 预训练权重

| 权重 | 说明 |
|------|------|
| `yolo11n.pt` | 入门推荐，最快最小 |
| `yolo11s/m/l/x.pt` | 精度更高、更慢 |
| `yolo26n.pt` 等 | 新一代检测权重（可选） |

COCO 预训练约 80 类（person、car、dog…）。

## 4. 输出字段（推理）

```text
result.boxes.xyxy   # 框坐标
result.boxes.conf   # 置信度
result.boxes.cls    # 类别 id
result.names        # id -> 名称
```

## 5. 入门步骤

1. 安装依赖并升级：`pip install -U ultralytics`
2. 阅读 [01 基础概念](../01-object-detection-basics.md)（IoU / NMS / mAP）
3. 运行：

```bash
python scripts/tasks/predict_detect.py
# 或
python scripts/01_predict_image.py --conf 0.25
```

4. 调整 `--conf` / `--iou`，观察框变多或变少  
5. 用 `scripts/04_train_coco8.py` 跑通训练  
6. 按 [03 数据集](../03-dataset-and-annotation.md) 准备自己的检测数据  

## 6. 迷你训练

```bash
python scripts/04_train_coco8.py --epochs 10
# 等价于: yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10
```

## 7. 评价指标

- **mAP50** / **mAP50-95**（框）
- Precision / Recall

## 8. 常见坑

- `conf` 太高 → 漏检；太低 → 误检  
- 类别不在 COCO → 必须自己训练  
- 小目标：适当增大 `imgsz`，或增加小目标样本  

## 9. 参考

- 官方：[Object Detection](https://docs.ultralytics.com/tasks/detect/)
