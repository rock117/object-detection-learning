# 任务 05 · 图像分类（Classify）

## 1. 它是什么

给 **整张图像** 预测一个（或 Top-K）类别，**没有框、没有 mask**。

回答的是：「这张图总体属于哪一类？」而不是「图里有哪些物体在哪」。

## 2. 适用场景

| 适合 | 不太适合 |
|------|----------|
| 场景分类（室内/室外）、品种识别 | 一张图多个目标要分别定位 |
| 工业「良品/不良」整图判定 | 需要缺陷在图中的位置 |
| 检索、相册自动打标签 | 需要像素级区域 |

若既要类别又要位置 → 用 **Detect**；只要整图标签 → **Classify** 更轻。

## 3. 预训练权重

| 权重 | 数据 |
|------|------|
| `yolo11n-cls.pt` | ImageNet 1000 类（入门） |
| `yolo11s/m/l/x-cls.pt` | 更大模型 |
| `yolo26n-cls.pt` 等 | 新一代可选 |

分类常用较小输入（如 **224**）。

## 4. 输出字段

```text
result.probs.top1        # 最高分类 id
result.probs.top1conf    # 对应置信度
result.probs.top5        # Top-5 id
result.names             # id -> 名称
```

## 5. 入门步骤

1. `pip install -U ultralytics`
2. 运行：

```bash
python scripts/tasks/predict_classify.py
```

3. 看 Top-5 类别是否合理（ImageNet 类别名可能很细）  
4. （进阶）自建分类数据目录结构：

```text
dataset/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

```bash
yolo classify train data=path/to/dataset model=yolo11n-cls.pt epochs=20 imgsz=224
```

## 6. 评价指标

- Top-1 / Top-5 Accuracy  

## 7. 常见坑

- 用检测权重去做分类任务（应使用 `-cls`）  
- 类别与 ImageNet 语义不符却期望直接可用 → 需微调  
- 目标很小且背景复杂 → 分类容易被背景主导，考虑先检测再裁剪分类  

## 8. 参考

- 官方：[Classification](https://docs.ultralytics.com/tasks/classify/)
