# 任务 02 · 实例分割（Instance Segmentation）

## 1. 它是什么

在检测框之外，再为 **每一个物体实例** 预测一块二值 mask（轮廓区域）。

同一类的多个物体会得到 **多块独立 mask**（人A ≠ 人B）。

## 2. 适用场景

| 适合 | 不太适合 |
|------|----------|
| 抠图、换背景、实例计数 | 只要框够用（检测更轻） |
| 缺陷区域面积、零件轮廓 | 只要「路面/天空」等区域标签（用语义分割） |
| 跟踪时需要更贴合的外形 | 极致实时且算力极紧（检测更快） |

## 3. 与语义分割的区别

| | 实例分割 | 语义分割 |
|--|----------|----------|
| 区分个体 | ✅ | ❌ |
| 输出 | `result.masks`（N 张） | `result.semantic_mask`（1 张类别图） |
| 框与置信度 | 通常有 | 通常无（逐像素类别） |
| 权重后缀 | `-seg` | `-sem`（YOLO26） |

## 4. 预训练权重

| 权重 | 说明 |
|------|------|
| `yolo11n-seg.pt` | 入门推荐 |
| `yolo11s/m/l/x-seg.pt` | 更大模型 |
| `yolo26n-seg.pt` 等 | 新一代可选 |

## 5. 输出字段

```text
result.boxes          # 同检测
result.masks.data     # (N,H,W) 实例 mask
result.masks.xy       # 多边形点（可视化/导出常用）
```

## 6. 入门步骤

1. `pip install -U ultralytics`
2. 运行：

```bash
python scripts/tasks/predict_segment.py
```

3. 打开 `runs/segment/` 看彩色 mask 叠加图  
4. 打印每个实例的面积（脚本内有示例）  
5. （进阶）准备多边形标注，数据格式与检测类似，但 label 含多边形点；可用 `coco8-seg.yaml` 试训：

```bash
yolo segment train data=coco8-seg.yaml model=yolo11n-seg.pt epochs=10
```

## 7. 评价指标

- Box mAP + **Mask mAP**（mask 更重要）

## 8. 常见坑

- 用错权重：`yolo11n.pt` 没有 mask  
- 标注成语义 PNG 却按实例格式训练 → 读 [03-semantic](./03-semantic-segment.md)  
- mask 锯齿：提高 `imgsz` 再比一次  

## 9. 参考

- 官方：[Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)
