# 目标检测学习项目（YOLO 多任务）

从零学习计算机视觉任务，以 **Ultralytics YOLO** 为主线：检测、实例分割、语义分割、深度估计、分类、姿态估计、旋转框（OBB）。

## 你能学到什么

1. 目标检测基础（bbox、IoU、NMS、mAP）
2. YOLO 七大任务的差异、场景与输出字段
3. 预训练推理（图片 / 视频 / 摄像头）
4. 自定义数据与训练入门
5. 导出与部署起步

## 环境准备

建议 Python 3.9+：

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -U ultralytics   # 语义分割 / 深度估计需要支持 YOLO26 的版本
```

有 NVIDIA GPU 时，按 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA 版 `torch`。

## 项目结构

```text
object-detection-learning/
├── README.md
├── requirements.txt
├── docs/
│   ├── 01-object-detection-basics.md
│   ├── 02-yolo-overview.md
│   ├── 03-dataset-and-annotation.md
│   ├── 04-learning-roadmap.md
│   ├── 05-yolo-tasks-overview.md      # 七大任务总览
│   └── tasks/                         # 各任务详细说明
│       ├── 01-detect.md
│       ├── 02-instance-segment.md
│       ├── 03-semantic-segment.md
│       ├── 04-depth.md
│       ├── 05-classify.md
│       ├── 06-pose.md
│       └── 07-obb.md
├── configs/
├── scripts/
│   ├── 01_predict_image.py            # 检测（基础）
│   ├── 02_predict_video.py
│   ├── 03_predict_webcam.py
│   ├── 04_train_coco8.py
│   ├── 05_export_model.py
│   └── tasks/                         # 七大任务入门推理
│       ├── predict_detect.py
│       ├── predict_segment.py
│       ├── predict_semantic.py
│       ├── predict_depth.py
│       ├── predict_classify.py
│       ├── predict_pose.py
│       └── predict_obb.py
├── data/
└── notebooks/
```

## 七大任务速查

| 任务 | 默认权重 | 文档 | 命令 |
|------|----------|------|------|
| 检测 | `yolo11n.pt` | [docs/tasks/01-detect.md](docs/tasks/01-detect.md) | `python scripts/tasks/predict_detect.py` |
| 实例分割 | `yolo11n-seg.pt` | [docs/tasks/02-instance-segment.md](docs/tasks/02-instance-segment.md) | `python scripts/tasks/predict_segment.py` |
| 语义分割 | `yolo26n-sem.pt` | [docs/tasks/03-semantic-segment.md](docs/tasks/03-semantic-segment.md) | `python scripts/tasks/predict_semantic.py` |
| 深度估计 | `yolo26n-depth.pt` | [docs/tasks/04-depth.md](docs/tasks/04-depth.md) | `python scripts/tasks/predict_depth.py` |
| 分类 | `yolo11n-cls.pt` | [docs/tasks/05-classify.md](docs/tasks/05-classify.md) | `python scripts/tasks/predict_classify.py` |
| 姿态估计 | `yolo11n-pose.pt` | [docs/tasks/06-pose.md](docs/tasks/06-pose.md) | `python scripts/tasks/predict_pose.py` |
| OBB | `yolo11n-obb.pt` | [docs/tasks/07-obb.md](docs/tasks/07-obb.md) | `python scripts/tasks/predict_obb.py` |

完整对比与选型：[docs/05-yolo-tasks-overview.md](docs/05-yolo-tasks-overview.md)

## 5 分钟上手（检测）

```bash
python scripts/01_predict_image.py
python scripts/04_train_coco8.py
python scripts/05_export_model.py
```

## 推荐学习顺序

1. [docs/01](docs/01-object-detection-basics.md) → 跑检测脚本  
2. [docs/05 任务总览](docs/05-yolo-tasks-overview.md) → 按兴趣跑 `scripts/tasks/`  
3. 精读对应 `docs/tasks/*.md`（场景、步骤、指标、坑）  
4. [docs/03](docs/03-dataset-and-annotation.md) → 自有小数据训练  
5. 按 [docs/04 路线图](docs/04-learning-roadmap.md) 做完整小项目  

## 模型规模参考

| 后缀 | 速度 | 精度 | 用途 |
|------|------|------|------|
| n | 最快 | 较低 | 入门 / CPU / 边缘 |
| s | 快 | 中 | 日常实验 |
| m | 中 | 较高 | 精度优先 |
| l / x | 慢 | 高 | 服务器 |

## 参考链接

- [Ultralytics 文档](https://docs.ultralytics.com/)
- [Tasks](https://docs.ultralytics.com/tasks/)
- [YOLO11](https://docs.ultralytics.com/models/yolo11/) / [YOLO26](https://docs.ultralytics.com/models/yolo26/)

## License

学习用途。第三方权重与数据集请遵守各自许可协议。
