# 任务 07 · 旋转框检测（OBB）

## 1. 它是什么

OBB = **Oriented Bounding Box**：预测带 **旋转角** 的矩形框，比水平框更贴合倾斜、细长目标。

输出一般是中心、宽高、角度（或四点坐标），外加类别与置信度。

## 2. 适用场景

| 适合 | 不太适合 |
|------|----------|
| 航拍/遥感：船只、车辆、飞机 | 普通监控里目标近似直立（水平框够用） |
| 倾斜文本行、条形零件 | 需要像素轮廓（用分割） |
| 仓储中任意朝向的长条货物 | 圆形目标（旋转收益有限） |

水平框对倾斜细长物体往往 **框住大量背景**，OBB 更紧凑。

## 3. 预训练权重

| 权重 | 说明 |
|------|------|
| `yolo11n-obb.pt` | 入门推荐（如 DOTA 等遥感风格预训练） |
| `yolo11s/m/l/x-obb.pt` | 更大模型 |
| `yolo26n-obb.pt` 等 | 新一代可选 |

类别与通用 COCO 检测 **不同**，预训练类名偏航拍场景；自定义场景请微调。

## 4. 输出字段

```text
result.obb.xyxyxyxy   # 四个顶点
result.obb.xywhr      # 中心、宽高、角度
result.obb.conf
result.obb.cls
```

## 5. 入门步骤

1. `pip install -U ultralytics`
2. 运行：

```bash
python scripts/tasks/predict_obb.py
```

3. 观察旋转框是否贴合物体长边  
4. 与 `predict_detect.py` 同图对比水平框差异  
5. （进阶）迷你训练：

```bash
yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=10
```

标签格式为旋转框（见官方 OBB 数据集说明），与普通 `xywh` 检测标签不同。

## 6. 评价指标

- OBB mAP（基于旋转 IoU）  

## 7. 常见坑

- 用水平框数据集直接训 OBB → 标签格式不匹配  
- 角度周期（0~π 等）导致训练不稳：跟官方数据规范走  
- 期望 COCO 人车类 → OBB 预训练类不同，需自训  

## 8. 参考

- 官方：[OBB](https://docs.ultralytics.com/tasks/obb/)
