# 04 · 学习路线图

按周推进即可；时间紧可压缩为几天冲刺。

---

## 第 1 阶段：跑通推理（0.5～1 天）

- [ ] 创建虚拟环境并安装 `requirements.txt`
- [ ] 阅读 [01 基础概念](./01-object-detection-basics.md)
- [ ] 运行 `scripts/01_predict_image.py`
- [ ] 改 `conf` / `iou`，观察框数量变化
- [ ] （可选）`02_predict_video.py` / `03_predict_webcam.py`

**验收**：能解释结果图上每个框的类别与置信度含义。

---

## 第 2 阶段：理解 YOLO 用法（1 天）

- [ ] 阅读 [02 YOLO 概览](./02-yolo-overview.md)
- [ ] 对比 `yolo11n` 与 `yolo11s` 速度/观感差异
- [ ] 阅读 `results` 打印信息，写一小段笔记

**验收**：能独立写出 10 行以内的 predict 代码。

---

## 第 3 阶段：跑通训练（1～2 天）

- [ ] 运行 `scripts/04_train_coco8.py`
- [ ] 打开 `runs/detect/train/` 看 `results.png`、权重 `best.pt`
- [ ] 用 `best.pt` 再跑一次推理
- [ ] 阅读 [03 数据集与标注](./03-dataset-and-annotation.md)

**验收**：知道 `data.yaml`、`epochs`、`batch`、`imgsz` 各控制什么。

---

## 第 3.5 阶段：读懂「真实标注」再开标（强烈建议）

- [ ] 阅读 [06 · 标注实战](./06-annotation-in-practice.md)
- [ ] 浏览 [术语表](./glossary.md)（至少搞清 SAM / SAM2 / 预标注 / 主动学习）
- [ ] 写一页自己的 Label Spec（类别、遮挡、最小框）

**验收**：能说清「为什么不要一上来手绘全部分割轮廓」。

---

## 第 4 阶段：自有小项目（3～7 天）

选题示例（选一个即可）：

- 桌上物品（杯子 / 手机 / 键盘）
- 宠物检测
- 交通小场景（车 / 人）

步骤：

1. 采集 100+ 张图（手机即可）  
2. 标注并划分 train/val  
3. 写自己的 `configs/my_data.yaml`  
4. 复制训练脚本，改 `data` 与 `project/name`  
5. 记录不同 `epochs`、模型大小下的 mAP  

**验收**：有一份「数据 → 训练 → 验证图」可复现实验记录。

---

## 第 5 阶段：多任务扩展（3～7 天）

先读 [05 任务总览](./05-yolo-tasks-overview.md)，再按兴趣选做（每个任务都遵循「文档 → 推理脚本 → 看输出字段」）：

- [ ] 实例分割：`docs/tasks/02` + `scripts/tasks/predict_segment.py`
- [ ] 姿态估计：`docs/tasks/06` + `scripts/tasks/predict_pose.py`
- [ ] 分类：`docs/tasks/05` + `scripts/tasks/predict_classify.py`
- [ ] OBB：`docs/tasks/07` + `scripts/tasks/predict_obb.py`
- [ ] 语义分割（需 YOLO26）：`docs/tasks/03` + `predict_semantic.py`
- [ ] 深度估计（需 YOLO26）：`docs/tasks/04` + `predict_depth.py`

**验收**：能说清「实例分割 vs 语义分割」以及「检测 vs OBB」各自何时用。

---

## 第 6 阶段：导出与部署（按兴趣）

- [ ] `scripts/05_export_model.py` 导出 ONNX  
- [ ] 了解 TensorRT / OpenVINO（选平台深入）  
- [ ] 阅读 Ultralytics 官方「Train / Val / Predict / Export」章节  

---

## 调试口诀

| 现象 | 优先检查 |
|------|----------|
| 完全检不到 | conf 是否过高；类别是否训过；输入路径 |
| 框乱飘 | 标签坐标是否未归一化；宽高是否写反 |
| mAP 极低 | train/val 路径；类别数；标注漏标 |
| CUDA OOM | 减小 `batch` / `imgsz`；换 n/s 模型 |
| 过拟合 | 加数据；降 epochs；用更小模型 |

---

## 下一步建议

练熟检测后，可按兴趣分支：

1. **跟踪**：ByteTrack 等，把框串成轨迹  
2. **分割**：像素级 mask  
3. **部署**：ONNX Runtime / 边缘设备  

保持原则：**小数据先跑通闭环，再追求指标**。
