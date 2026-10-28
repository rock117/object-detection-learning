# 05 · YOLO 多任务总览

Ultralytics YOLO 把多种视觉任务统一到同一套 API：`YOLO(权重).predict / train / val / export`。  
选任务 = **选对权重后缀 + 理解输出字段**。

---

## 1. 七大任务一眼对照

| 任务 | 权重示例 | 主要输出 | 核心问题 |
|------|----------|----------|----------|
| **检测 Detect** | `yolo11n.pt` | 水平框 + 类别 | 物体在哪、是什么 |
| **实例分割 Segment** | `yolo11n-seg.pt` | 框 + **每个实例** mask | 每个物体的精确轮廓 |
| **语义分割 Semantic** | `yolo26n-sem.pt` | **整图像素类别图** | 每个像素属于哪一类（不区分个体） |
| **深度估计 Depth** | `yolo26n-depth.pt` | 每像素深度（米） | 离相机有多远 |
| **分类 Classify** | `yolo11n-cls.pt` | 整图类别 | 这张图整体是什么 |
| **姿态估计 Pose** | `yolo11n-pose.pt` | 框 + 关键点 | 人的骨架 / 关节点 |
| **旋转框 OBB** | `yolo11n-obb.pt` | 带角度的框 | 倾斜/细长物体怎么框 |

> **版本说明**  
> - Detect / Segment / Classify / Pose / OBB：YOLO11 与 YOLO26 都有预训练权重。本仓库入门默认 **YOLO11**。  
> - Semantic / Depth：当前官方预训练以 **YOLO26** 为主（`-sem` / `-depth`）。请先 `pip install -U ultralytics`。

---

## 2. 怎么选任务（场景导向）

| 你的需求 | 优先任务 |
|----------|----------|
| 数人数、找车辆、安防报警 | Detect |
| 抠图、量面积、实例级测量 | Segment（实例） |
| 可行驶区域、地表覆盖、场景解析 | Semantic |
| 避障距离、AR、粗糙 3D 布局 | Depth |
| 整图品类识别、质检「有无缺陷类」 | Classify |
| 健身计数、跌倒检测、动作分析 | Pose |
| 航拍船只/车辆、倾斜文本、遥感 | OBB |

**实例分割 vs 语义分割（必读）**

- 图里有 3 辆车：  
  - **实例**：3 块独立 mask（车1/车2/车3）  
  - **语义**：所有车像素都是一类 `car`，不分开个体  

---

## 3. 统一入门节奏（每种任务都适用）

1. **读对应文档**（本目录 `docs/tasks/`）  
2. **跑预训练推理**（`scripts/tasks/`）  
3. **看结果字段**（框 / mask / 关键点 / 深度图）  
4. **（可选）迷你集 train** 验证自定义数据格式  
5. **换自己的小数据** 再谈指标与部署  

---

## 4. 代码与文档索引

| 任务 | 文档 | 入门脚本 |
|------|------|----------|
| Detect | [tasks/01-detect.md](./tasks/01-detect.md) | `scripts/01_predict_image.py` 或 `scripts/tasks/predict_detect.py` |
| Instance Seg | [tasks/02-instance-segment.md](./tasks/02-instance-segment.md) | `scripts/tasks/predict_segment.py` |
| Semantic Seg | [tasks/03-semantic-segment.md](./tasks/03-semantic-segment.md) | `scripts/tasks/predict_semantic.py` |
| Depth | [tasks/04-depth.md](./tasks/04-depth.md) | `scripts/tasks/predict_depth.py` |
| Classify | [tasks/05-classify.md](./tasks/05-classify.md) | `scripts/tasks/predict_classify.py` |
| Pose | [tasks/06-pose.md](./tasks/06-pose.md) | `scripts/tasks/predict_pose.py` |
| OBB | [tasks/07-obb.md](./tasks/07-obb.md) | `scripts/tasks/predict_obb.py` |

---

## 5. 标注与术语

- [06 · 真实项目标注实战](./06-annotation-in-practice.md)（预标注、SAM、外包、30 天计划）  
- [术语表](./glossary.md)（SAM / SAM2、主动学习、伪标签等）

## 6. 官方文档

- [Tasks 总览](https://docs.ultralytics.com/tasks/)  
- [YOLO11](https://docs.ultralytics.com/models/yolo11/) / [YOLO26](https://docs.ultralytics.com/models/yolo26/)
