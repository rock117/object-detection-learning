# 简介

从零学习计算机视觉任务，以 **Ultralytics YOLO** 为主线：检测、实例分割、语义分割、深度估计、分类、姿态估计、旋转框（OBB）。

本站由仓库内 `docs/` 通过 [mdBook](https://rust-lang.github.io/mdBook/) 生成；代码与脚本仍在 [GitHub 仓库](https://github.com/rock117/object-detection-learning)。

## 你能学到什么

1. 目标检测基础（bbox、IoU、NMS、mAP）
2. YOLO 七大任务的差异、场景与输出字段
3. 预训练推理（图片 / 视频 / 摄像头）
4. 自定义数据与训练入门
5. 导出与部署起步

## 环境准备

建议 Python 3.9+。在仓库根目录：

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -U ultralytics   # 语义分割 / 深度估计需要支持 YOLO26 的版本
```

有 NVIDIA GPU 时，按 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA 版 `torch`。

## 本地预览本文档站

已安装 [mdBook](https://github.com/rust-lang/mdBook) 时，在仓库的 `docs/` 目录：

```bash
cd docs
mdbook serve --open
```

浏览器打开 `http://localhost:3000`。只构建静态站：`mdbook build`（输出到仓库根目录 `book/`）。

## 5 分钟上手（检测）

在仓库根目录：

```bash
python scripts/01_predict_image.py
python scripts/04_train_coco8.py
python scripts/05_export_model.py
```

## 推荐阅读顺序

1. [01 · 目标检测基础](./01-object-detection-basics.md) → 跑检测脚本
2. [05 · 任务总览](./05-yolo-tasks-overview.md) → 按兴趣跑 `scripts/tasks/`
3. 精读对应 [多任务](./05-yolo-tasks-overview.md) 章节
4. [06 · 标注实战](./06-annotation-in-practice.md) + [术语表](./glossary.md)
5. [03 · 数据集与标注](./03-dataset-and-annotation.md) → 自有小数据训练
6. 按 [04 · 学习路线图](./04-learning-roadmap.md) 做完整小项目
7. （可选）[07 · ByteTrack](./07-bytetrack.md)：`python scripts/06_track_bytetrack.py`

## 七大任务速查

| 任务 | 默认权重 | 文档 | 命令 |
|------|----------|------|------|
| 检测 | `yolo11n.pt` | [检测](./tasks/01-detect.md) | `python scripts/tasks/predict_detect.py` |
| 实例分割 | `yolo11n-seg.pt` | [实例分割](./tasks/02-instance-segment.md) | `python scripts/tasks/predict_segment.py` |
| 语义分割 | `yolo26n-sem.pt` | [语义分割](./tasks/03-semantic-segment.md) | `python scripts/tasks/predict_semantic.py` |
| 深度估计 | `yolo26n-depth.pt` | [深度估计](./tasks/04-depth.md) | `python scripts/tasks/predict_depth.py` |
| 分类 | `yolo11n-cls.pt` | [分类](./tasks/05-classify.md) | `python scripts/tasks/predict_classify.py` |
| 姿态估计 | `yolo11n-pose.pt` | [姿态估计](./tasks/06-pose.md) | `python scripts/tasks/predict_pose.py` |
| OBB | `yolo11n-obb.pt` | [OBB](./tasks/07-obb.md) | `python scripts/tasks/predict_obb.py` |

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
