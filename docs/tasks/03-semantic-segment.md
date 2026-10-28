# 任务 03 · 语义分割（Semantic Segmentation）

## 1. 它是什么

为 **每个像素** 分配一个类别 ID，得到整图稠密类别图（class map）。

- 不区分「第几辆车」  
- 同类像素合并成同一区域  

> 需要 **YOLO26** 语义权重（`-sem`）。请使用较新的 Ultralytics：`pip install -U ultralytics`。

## 2. 适用场景

| 适合 | 不太适合 |
|------|----------|
| 自动驾驶：路面、人行道、植被、天空 | 要数「有几个人」 |
| 遥感土地覆盖、医疗器官区域 | 要抠出单个商品实例 |
| 场景解析、布局理解 | 只要整图一个标签（用分类） |

## 3. 预训练权重

| 权重 | 预训练数据倾向 |
|------|----------------|
| `yolo26n-sem.pt` | Cityscapes 城市场景（入门常用） |
| `yolo26n-sem-ade20k.pt` | ADE20K 更多场景类别 |
| `yolo26s/m/l/x-sem.pt` | 更大模型 |

Cityscapes 类推理建议较大 `imgsz`（如 1024）；ADE20K 常用 640。

## 4. 输出字段

```text
result.semantic_mask.data   # (H,W) 整型类别图
# 注意：通常没有 result.boxes / result.masks（实例）
```

## 5. 入门步骤

1. 升级库：`pip install -U ultralytics`
2. 运行：

```bash
python scripts/tasks/predict_semantic.py
```

3. 查看保存的可视化；脚本会统计各类像素占比  
4. 对比同一张图用 `predict_segment.py` 的差异（个体 vs 区域）  
5. （进阶）迷你训练：

```bash
yolo semantic train data=cityscapes8.yaml model=yolo26n-sem.pt epochs=20 imgsz=1024
```

自定义数据：单通道 PNG mask，像素值=类别 ID，`255` 为 ignore。

## 6. 评价指标

- **mIoU**（各类 IoU 平均）  
- Pixel Accuracy  

## 7. 常见坑

- YOLO11 **没有** 官方 `-sem` 预训练；请用 YOLO26  
- 边界锯齿：增大 `imgsz` 再推理 `.pt` 模型  
- 与实例分割数据格式不同：语义常用 PNG mask  

## 8. 参考

- 官方：[Semantic Segmentation](https://docs.ultralytics.com/tasks/semantic/)
