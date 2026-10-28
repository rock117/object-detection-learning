# 任务 04 · 单目深度估计（Depth）

## 1. 它是什么

从 **一张 RGB 图** 预测每个像素到相机的大致距离（单位：米），得到稠密深度图。

不是双目立体，也不直接输出物体类别框（可用检测模型另开一路）。

> 需要 **YOLO26** 深度权重（`-depth`）：`pip install -U ultralytics`。

## 2. 适用场景

| 适合 | 不太适合 |
|------|----------|
| 机器人近距避障、通行性分析 | 需要毫米级测绘精度 |
| AR/VR 粗略空间布局 | 必须有度量标定的工业测量（需标定/专用传感） |
| 辅助 3D 理解、相对远近可视化 | 只要类别标签（用检测/分割） |

深度是 **相对/估计值**，跨场景尺度可能有偏差，工程上常做校准或与其他传感器融合。

## 3. 预训练权重

| 权重 | 说明 |
|------|------|
| `yolo26n-depth.pt` | 入门推荐 |
| `yolo26s/m/l/x-depth.pt` | 更大更准更慢 |

官方常用输入尺度约 **768**（可按显存调整）。

## 4. 输出字段

```text
result.depth.data   # (H,W) float，单位米
```

可视化常用伪彩色（近冷/远暖或反过来，按习惯）。

## 5. 入门步骤

1. `pip install -U ultralytics`
2. 运行：

```bash
python scripts/tasks/predict_depth.py
```

3. 查看深度热力图与最小/最大/平均深度统计  
4. 换室内外不同照片，感受尺度变化  
5. （进阶）迷你训练：

```bash
yolo depth train data=depth8.yaml model=yolo26n-depth.pt epochs=20 imgsz=640
```

## 6. 评价指标（了解）

常见有 AbsRel、RMSE、δ1（delta1）等；不同数据集（如 NYU）报告方式不同。

## 7. 常见坑

- 当成「激光雷达真值」直接用于安全关键决策  
- 旧版 ultralytics 无 depth 任务 → 升级到支持 YOLO26 的版本  
- 与检测结果对齐时注意分辨率与预处理一致  

## 8. 参考

- 官方：[Depth Estimation](https://docs.ultralytics.com/tasks/depth/)
