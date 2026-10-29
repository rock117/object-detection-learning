# 07 · ByteTrack 多目标跟踪

> **和 YOLO 的关系**：ByteTrack **不是** YOLO 网络结构的一部分，而是接在检测（或分割 / 姿态）之后的 **多目标跟踪（MOT）** 算法。  
> Ultralytics 已内置，用 `model.track(..., tracker="bytetrack.yaml")` 即可。

练熟 [检测](./tasks/01-detect.md) 后，若要从「每一帧有哪些框」进到「同一物体跨帧是同一个 ID」，优先读本章。

---

## 1. 它是什么

**检测**回答：这一帧里，物体在哪、是什么。  
**跟踪**回答：这些框里，哪些是同一辆车 / 同一个人，并给每个目标一个稳定的 **track id**。

| 对比 | `predict`（检测） | `track` + ByteTrack |
|------|-------------------|---------------------|
| 输入 | 图 / 视频均可 | **视频或摄像头流**（跨帧才有意义） |
| 输出 | 框 + 类 + 置信度 | 再加 **`id`**（轨迹编号） |
| 状态 | 帧间独立 | 帧间关联，有短期记忆 |

ByteTrack（Zhang et al., ECCV 2022）的核心思路：

1. **第一阶段**：用高置信度检测框，和已有轨迹做匹配（IoU + 卡尔曼预测）。
2. **第二阶段**：把「没配上的轨迹」再拿去和 **低置信度** 检测框匹配，短暂遮挡时尽量把 ID 救回来。
3. **不做**外观 ReID、**不做**相机运动补偿 → 轻量、快，适合固定机位。

```text
视频帧 → YOLO Detect/Seg/Pose → 检测框
                              ↓
                         ByteTrack 关联
                              ↓
                    带 track id 的框（轨迹）
```

---

## 2. 使用场景

| 适合 | 不太适合 / 需换方案 |
|------|---------------------|
| 固定摄像头：人流、车流、过线计数 | 手持晃动、无人机剧烈运动（优先 BoT-SORT / 带 GMC 的跟踪器） |
| 安防：同一目标连续报警、停留时长 | 长时间完全遮挡后「认人」（需要 ReID） |
| 产线：传送带上同 ID 质检跟拍 | 目标外观极相似且交叉频繁（易 ID 互换） |
| 姿态之上：同一人骨架跨帧（健身计数） | 单张图片分析（用 `predict` 即可） |
| 标注辅助：视频里框住后少标几帧 | 要求司法级身份一致性 |

**选型速记（Ultralytics 内置）**

| 跟踪器 | 配置 | 何时选 |
|--------|------|--------|
| **ByteTrack** | `bytetrack.yaml` | 固定机位、要速度、入门首选 |
| BoT-SORT | `botsort.yaml` | 相机在动；可开 ReID |
| TrackTrack 等 | 见官方 Track 文档 | 更复杂遮挡 / 多线索关联 |

官方对照：[Multi-Object Tracking](https://docs.ultralytics.com/modes/track/)

---

## 3. 环境与前置

1. 已能跑通检测：`python scripts/01_predict_image.py`
2. `pip install -U ultralytics`（跟踪随库附带，无需另装 ByteTrack）
3. 准备一段视频放到 `data/videos/`，或用摄像头 `0`

---

## 4. 最短可运行例子

### 4.1 CLI

```bash
yolo track model=yolo11n.pt source=data/videos/demo.mp4 tracker=bytetrack.yaml
```

### 4.2 Python（推荐）

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # 也可用 *-seg / *-pose / 自训 best.pt

results = model.track(
    source="data/videos/demo.mp4",
    tracker="bytetrack.yaml",
    persist=True,   # 跨调用保持轨迹；stream / 循环读帧时尤其重要
    conf=0.25,
    save=True,
    stream=True,    # 逐帧迭代，视频别一次塞进内存
)

for r in results:
    if r.boxes is None or r.boxes.id is None:
        continue
    for box, tid in zip(r.boxes, r.boxes.id):
        cls_id = int(box.cls[0])
        print(f"id={int(tid)}  {r.names[cls_id]}  conf={float(box.conf[0]):.2f}")
```

### 4.3 本仓库脚本

```bash
# 本地视频
python scripts/06_track_bytetrack.py --source data/videos/demo.mp4

# 摄像头（Windows 常见为 0）
python scripts/06_track_bytetrack.py --source 0 --show

# 只跟踪人（COCO person=0）
python scripts/06_track_bytetrack.py --source data/videos/demo.mp4 --classes 0

# 自定义阈值配置
python scripts/06_track_bytetrack.py --source data/videos/demo.mp4 --tracker configs/bytetrack.example.yaml
```

结果默认写到 `runs/track/bytetrack/`。

---

## 5. 输出字段

在检测字段之上，多了轨迹 ID：

```text
result.boxes.xyxy   # 框
result.boxes.conf   # 置信度
result.boxes.cls    # 类别
result.boxes.id     # track id（Tensor）；本帧无跟踪时可能为 None
```

判断本帧是否有有效 ID：

```python
if r.boxes is not None and r.boxes.id is not None:
    ids = r.boxes.id.int().tolist()
```

可视化时 Ultralytics 会在框旁画上 id；业务侧常用 `id` 做：

- 去重计数（同一人过线只计一次）
- 轨迹点列表（中心点随时间变化）
- 按 id 聚合停留帧数 / 速度粗估

---

## 6. 实战小项目：按 ID 过线计数（示意）

固定机位、画面中间有一条水平线时，可判断「中心点从线上到线下」算进入一次：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
line_y = 360  # 按你的分辨率改
last_side = {}  # id -> "above" | "below"
enter_count = 0

for r in model.track(
    source="data/videos/demo.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    classes=[0],  # 只跟人
    stream=True,
    verbose=False,
):
    if r.boxes is None or r.boxes.id is None:
        continue
    xyxy = r.boxes.xyxy.cpu()
    ids = r.boxes.id.int().cpu().tolist()
    for box, tid in zip(xyxy, ids):
        cx = float((box[0] + box[2]) / 2)
        cy = float((box[1] + box[3]) / 2)
        side = "above" if cy < line_y else "below"
        prev = last_side.get(tid)
        if prev == "above" and side == "below":
            enter_count += 1
            print(f"id={tid} 进入，累计={enter_count}")
        last_side[tid] = side
```

要点：计数依赖 **稳定 id**；若频繁换号，计数会偏大 → 先把检测和跟踪阈值调稳，再上业务逻辑。

更完整的读写与 CLI 封装见 `scripts/06_track_bytetrack.py`（含按 id 汇总出现帧数）。

---

## 7. 摄像头 / OpenCV 循环（persist）

自己 `VideoCapture` 读帧时，**每次** `track` 都要 `persist=True`，否则每帧都是新轨迹：

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )
    plotted = results[0].plot()
    cv2.imshow("bytetrack", plotted)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

多路摄像头：**每个源一个 YOLO 实例**（或自行隔离 tracker 状态），不要共用一个 `persist` 状态去跟多路流。

---

## 8. 调参（`bytetrack.yaml`）

可直接改官方配置，或复制本仓库示例：[`configs/bytetrack.example.yaml`](../configs/bytetrack.example.yaml)。

| 参数 | 含义 | 调参直觉 |
|------|------|----------|
| `track_high_thresh` | 第一阶段高分框门槛 | 抬高 → 轨迹更干净，易断 |
| `track_low_thresh` | 第二阶段低分救援门槛 | 过低 → 易把噪声框接进轨迹 |
| `new_track_thresh` | 开新轨迹的最低分 | 抬高 → 少假轨迹；过低 → id 爆炸 |
| `track_buffer` | 丢失后保留多少帧 | 遮挡久可加大；过大易「幽灵」再匹配错 |
| `match_thresh` | 关联相似度门槛 | 和检测质量一起调 |
| `fuse_score` | 是否融合检测分做匹配 | 默认 `True`，弱检测时更稳 |

检测侧也要一起调：`conf` / `iou` / `imgsz`。跟踪救不了「检都检不出来」的模型。

---

## 9. 常见坑

| 现象 | 优先检查 |
|------|----------|
| 每帧 id 全变 | 是否漏了 `persist=True`；是否每帧新建了 `YOLO()` |
| id 狂涨、同一个人多个号 | `new_track_thresh` 过低；遮挡；`track_buffer` 过短 |
| 两人交叉后身份互换 | ByteTrack 无 ReID；试 BoT-SORT + ReID，或限制 `classes` |
| 画面一晃 id 全乱 | 换 `botsort.yaml`（相机运动补偿） |
| `boxes.id` 为 `None` | 该帧无成功关联；开头几帧或全被滤掉时正常 |
| 只要统计人数却跟了车 | `--classes 0`（或你的类别 id） |

---

## 10. 和分割 / 姿态一起用

跟踪器吃的是「框」；分割、姿态模型同样可以 `track`：

```python
YOLO("yolo11n-seg.pt").track(source="video.mp4", tracker="bytetrack.yaml", persist=True)
YOLO("yolo11n-pose.pt").track(source="video.mp4", tracker="bytetrack.yaml", persist=True)
```

实例 mask / 关键点会随框一起带上 id，便于「同一人的骨架序列」。

---

## 11. 学习验收

- [ ] 能说清：检测 vs 跟踪；ByteTrack 为何适合固定机位  
- [ ] 跑通 `scripts/06_track_bytetrack.py`，结果视频上能看到 id  
- [ ] 知道 `persist=True` 和 `boxes.id`  
- [ ] 能根据「id 乱飘 / 相机晃」决定调参或换 BoT-SORT  

---

## 12. 参考

- 论文：[ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)  
- Ultralytics：[Track mode](https://docs.ultralytics.com/modes/track/)  
- 本仓库检测入门：[tasks/01-detect.md](./tasks/01-detect.md)  
- 路线图中的位置：[04 · 学习路线图](./04-learning-roadmap.md)
