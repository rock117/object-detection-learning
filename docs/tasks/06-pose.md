# 任务 06 · 姿态估计（Pose / Keypoints）

## 1. 它是什么

在检测人的同时，预测一组 **人体关键点**（关节），并可连成骨架。

COCO 姿态常见 **17 个点**（鼻、眼、耳、肩、肘、腕、髋、膝、踝等）。每个点通常还有可见性/置信度。

## 2. 适用场景

| 适合 | 不太适合 |
|------|----------|
| 健身计数（深蹲、俯卧撑） | 只要「有没有人」（检测即可） |
| 跌倒检测、工位动作规范 | 手部精细 21 点（需专用手部模型） |
| 体感交互、简易动作分析 | 多人极度遮挡且要求很高精度 |

本仓库入门默认 **人体姿态**。通用「物体关键点」需按数据集自定义 kpt 配置。

## 3. 预训练权重

| 权重 | 说明 |
|------|------|
| `yolo11n-pose.pt` | 入门推荐 |
| `yolo11s/m/l/x-pose.pt` | 更大模型 |
| `yolo26n-pose.pt` 等 | 新一代可选 |

## 4. 输出字段

```text
result.boxes                # 人体框
result.keypoints.xy         # (N, K, 2) 像素坐标
result.keypoints.conf       # (N, K) 点置信度（若有）
```

## 5. 入门步骤

1. `pip install -U ultralytics`
2. 运行：

```bash
python scripts/tasks/predict_pose.py
```

3. 看骨架连线；打印关键点坐标  
4. 摄像头实时：

```bash
python scripts/tasks/predict_pose.py --source 0 --show
```

5. （进阶）迷你训练：

```bash
yolo pose train data=coco8-pose.yaml model=yolo11n-pose.pt epochs=10
```

自定义数据：在检测框标签基础上增加关键点坐标（见官方 Pose 数据集格式）。

## 6. 评价指标

- Pose mAP（对关键点 OKS 等匹配方式）  
- 同时可看 box mAP  

## 7. 常见坑

- 用普通 `yolo11n.pt` 期望出骨架 → 必须用 `-pose`  
- 侧身/遮挡导致点漂移：加难例数据或提高 conf 过滤低质量点  
- 多人 ID 不稳定：姿态之上再加跟踪（ByteTrack 等）  

## 8. 参考

- 官方：[Pose Estimation](https://docs.ultralytics.com/tasks/pose/)
