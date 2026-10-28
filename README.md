# 目标检测学习项目（YOLO）

从零学习目标检测，以 **Ultralytics YOLO** 为主线，边读文档边跑代码。

## 你能学到什么

1. 目标检测基本概念（bbox、IoU、NMS、mAP）
2. YOLO 系列思路与 YOLO11/YOLOv8 用法
3. 预训练模型推理（图片 / 视频 / 摄像头）
4. 自定义数据集标注与训练
5. 简单导出与部署入门

## 环境准备

建议 Python 3.9+，使用虚拟环境：

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

有 NVIDIA GPU 时，按 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA 版 `torch`，再装本项目依赖。CPU 也能跑推理和小数据集训练，只是会慢一些。

## 项目结构

```text
object-detection-learning/
├── README.md                 # 本文件
├── requirements.txt
├── docs/                     # 学习文档
│   ├── 01-object-detection-basics.md
│   ├── 02-yolo-overview.md
│   ├── 03-dataset-and-annotation.md
│   └── 04-learning-roadmap.md
├── configs/
│   └── coco8.yaml            # 迷你数据集配置示例
├── scripts/                  # 可运行入门代码
│   ├── 01_predict_image.py
│   ├── 02_predict_video.py
│   ├── 03_predict_webcam.py
│   ├── 04_train_coco8.py
│   └── 05_export_model.py
├── data/
│   ├── images/               # 放你的测试图片
│   └── videos/               # 放你的测试视频
├── weights/                  # 下载的模型权重（自动生成）
└── notebooks/                # 可选：Jupyter 实验
```

## 5 分钟上手

安装依赖后，在项目根目录执行：

```bash
# 1. 用预训练模型检测一张图（会自动下载 yolo11n.pt）
python scripts/01_predict_image.py

# 2. 视频推理（需先把视频放到 data/videos/）
python scripts/02_predict_video.py --source data/videos/demo.mp4

# 3. 摄像头实时检测
python scripts/03_predict_webcam.py

# 4. 用官方迷你集 coco8 跑通训练流程（几分钟）
python scripts/04_train_coco8.py

# 5. 导出 ONNX（便于后续部署）
python scripts/05_export_model.py
```

结果默认写到 `runs/` 目录。

## 推荐学习顺序

按 `docs/04-learning-roadmap.md` 走即可。简版：

1. 读 `docs/01-object-detection-basics.md`
2. 跑 `scripts/01_predict_image.py`，观察框与类别
3. 读 `docs/02-yolo-overview.md`
4. 跑训练脚本，理解 `epochs` / `imgsz` / `batch`
5. 读 `docs/03-dataset-and-annotation.md`，准备自己的小数据集
6. 改配置再训练，对比指标变化

## 常用模型体积参考（YOLO11）

| 模型 | 参数量量级 | 速度 | 精度 | 适合场景 |
|------|------------|------|------|----------|
| yolo11n | 最小 | 最快 | 较低 | 入门、CPU、边缘 |
| yolo11s | 小 | 快 | 中 | 日常实验 |
| yolo11m | 中 | 中 | 较高 | 精度优先 |
| yolo11l / x | 大 | 慢 | 高 | 服务器、追求 mAP |

入门建议先用 **yolo11n**。

## 参考链接

- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [YOLO11 介绍](https://docs.ultralytics.com/models/yolo11/)
- [COCO 数据集](https://cocodataset.org/)

## License

学习用途。第三方权重与数据集请遵守各自许可协议。
