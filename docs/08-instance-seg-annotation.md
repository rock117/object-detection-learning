# 08 · 实例分割标注（多边形）实战

目标：把 `datasets/fruit/images/raw/` 里的榴莲 / 哈密瓜图，标成 **YOLO-seg 多边形标签**，
并能通过质检、划分、训练。

> 检测（画框）标注见 [06 · 标注实战](./06-annotation-in-practice.md)；本篇只讲**多边形**。

---

## 1. 先想清楚：真的需要分割吗

| | 检测（框） | 实例分割（多边形） |
|--|-----------|-------------------|
| 标注速度 | 1×（约 2~4 秒/个） | **3~6×**（约 10~20 秒/个，成堆重叠更慢） |
| 标签行格式 | `cls xc yc w h`（5 个数） | `cls x1 y1 x2 y2 ...`（≥7 个数） |
| 能回答 | 有几个、在哪 | 有几个、在哪、**轮廓/面积/像素** |
| 典型用途 | 计数、报警、跟踪 | 抠图、大小分级、遮挡占比、精细抓取点 |

**结论**：只做「数榴莲」用检测就够了；要做「分割出每个榴莲的果皮区域 / 估算体积 / 抠图」，才上实例分割。

---

## 2. YOLO-seg 标签到底长什么样

一条标签 = 一个目标实例 = `类别 id + 归一化多边形顶点`：

```text
# labels/train/durian_0001.txt
0 0.312 0.204 0.402 0.198 0.455 0.262 0.421 0.351 0.330 0.360   # 榴莲 1
0 0.612 0.412 0.702 0.408 0.751 0.470 0.712 0.551                # 榴莲 2
1 0.120 0.700 0.210 0.705 0.240 0.780 0.150 0.812                # 哈密瓜 1
```

规则（踩错任何一条都会训练报错或学歪）：

- 坐标是 **相对宽高的 0~1 浮点**，不是像素
- 每行第一个数是类别 id，后面坐标个数必须是**偶数且 ≥6**（= 至少 3 个点）
- 一行 = 一个实例；同一类别多个物体 → 多行（这就是「实例」分割）
- **不能和检测框格式混在一个数据集里**，Ultralytics 会直接报：
  `Segment dataset requires equal numbers of boxes and segments`
- 多边形建议 ≥8 个点、顶点按顺序（顺/逆时针都行，但不能交叉）
- 没有目标的图 → **空 txt 文件**（背景图，建议留 5~10%）

---

## 3. 目录结构

```text
datasets/fruit/
├── images/
│   ├── raw/durian/*.jpg           # 原始图（已在）
│   ├── raw/hami_melon/*.jpg
│   ├── train/*.jpg                # split_dataset.py 生成
│   └── val/*.jpg
└── labels/
    ├── raw/durian/*.txt           # 标注工具导出的多边形标签（同名）
    ├── raw/hami_melon/*.txt
    ├── train/*.txt
    └── val/*.txt
```

配置：`configs/fruit-seg.yaml`（`names: 0 durian, 1 hami_melon`）

---

## 4. 工具选型

| 工具 | 多边形体验 | SAM 辅助 | 适合 | 安装 |
|------|-----------|----------|------|------|
| **X-AnyLabeling** ⭐推荐 | 好（拖点、删点、自动闭合） | ✅ 内置 SAM / SAM2 / YOLO | 单机、Windows、要 AI 预标注 | `pip install anylabeling` 或下载 release |
| **CVAT** | 很好（专业多边形工具） | ✅（Nuclio 部署 SAM） | 团队协作、要审核流程 | Docker 部署 |
| **labelme** | 一般（够用） | ❌ | 最轻量、纯手工 | `pip install labelme` |
| **Roboflow** | 好 | ✅ | 想省事、不介意上传数据 | 网页 |
| Label Studio | 差（多边形别扭） | 部分 | 已在用其做分类/检测 | `pip install label-studio` |

**给这个项目的建议**：`X-AnyLabeling` + SAM2，本地跑、离线、能边点边出 mask，比纯手画快 3 倍以上。

---

## 5. X-AnyLabeling 实操（推荐路线）

### 5.1 安装与准备

```bash
pip install anylabeling        # 或从 GitHub Release 下 exe
```

1. 启动后 `File → Change Output Dir`，把标签输出目录设成 **和图片同目录**（最省事），
   或统一设成 `datasets/fruit/labels/raw/durian`（注意要手动保持同名）。
2. `File → Open Dir` 打开 `datasets/fruit/images/raw/durian`。
3. 先建类别：`Tools → Label Manager`（或界面右侧），按顺序录入，**顺序必须与 yaml 一致**：

```text
id 0 → durian
id 1 → hami_melon
```

4. 开启多边形模式：左侧工具栏选 **Polygon**（快捷键 `P`）。

### 5.2 手画一个多边形（不用 AI）

`点顶点 → 点回起点闭合` → 弹出类别选择 → 选中 `durian`。
编辑：拖动顶点微调、在边上双击加点、选中点按 `Delete` 删点、
整体选中按方向键微移、`Ctrl+Z` 撤销。

### 5.3 用 SAM2 半自动（核心提速）

1. `Tools → AI Model`（或左上角"魔棒"图标），选择 `Segment Anything 2`，
   下载 `sam2.1_b.pt`（约 300MB，CPU 也能跑，GPU 更快）。
2. 交互方式：
   - **点一个点** → 出该物体 mask → 满意就按确认，自动转成多边形
   - **框一下**（Box）→ 出框内物体 mask，适合成堆里挑单个榴莲
   - 点错了按住 `Shift` 加点、`Ctrl`/右键 减点修正
3. 出来的 mask **仍然是多边形**，可以继续拖顶点修正 → 这就是"AI 预标注 + 人工精修"。

### 5.4 导出

`File → Export → Export Annotations` → 格式选 **`YOLO Segmentation`**（或 `YOLO`+多边形）。

- ⚠️ 千万不要选 `YOLO Detection`：那样只导出框（5 列），分割训练会直接报错
- 导出后建议用 `classes.txt` 里的顺序再核对一遍 id

### 5.5 快捷键（省一半时间）

| 快捷键 | 作用 |
|--------|------|
| `P` | 多边形模式 |
| `Ctrl+S` | 保存当前 |
| `D` / `A` | 下一张 / 上一张 |
| `Delete` | 删除选中标注 |
| `Ctrl+Z` / `Ctrl+Y` | 撤销 / 重做 |
| `Esc` | 取消当前绘制 |

---

## 6. CVAT 路线（团队/需要审核时）

1. `Create Task` → 上传 `raw/durian` 图片 → `Labels` 里添加 `durian`、`hami_melon`（顺序同 yaml）
2. 标注界面选 **Polygon**（不是 Rectangle）；密集场景用 `Z` 放大，
   用 `Draw new polygon (N)` 逐个描边
3. 有 GPU 时可部署 `Segment Anything` 交互式模型，点选生成 mask 再人工修
4. 导出：`Export task dataset → Ultralytics YOLO Segmentation 1.0`
   → 解压后 `labels/<split>/*.txt` 就是多边形格式，直接可用

---

## 7. 标注规范（先定规则，再动手）

针对榴莲 / 哈密瓜这种「成堆、有网纹、易重叠」的物体：

| 情形 | 规则建议 |
|------|----------|
| 遮挡 | 遮挡 <50% 照常描完整轮廓（脑补被遮部分）；>50% 不标或只标露出的部分（二选一并写进规范） |
| 成堆重叠 | **每个果实一条多边形**，边界按可见轮廓走，不要求遮挡处精确 |
| 贴边截断 | 只要主体在画面内就标，轮廓沿图像边界走 |
| 最小目标 | 短边 <20px 的不标（分割模型学不到） |
| 阴影/反光 | 不算物体，多边形只包果皮 |
| 切开/剖开的果肉 | 单独决策：算同类还是不算（写进规范，否则模型学乱） |
| 类别歧义 | 网纹瓜统一记 `hami_melon`（见采集报告说明） |

建议先写 1 页 `label_spec.md`，贴 3~5 张「正确/错误」示例图，后面省无数返工。

---

## 8. SAM 预标注脚本化（可选，批量用）

不用标注工具时，可以用 Ultralytics 的 SAM2 "everything" 模式先批量生成多边形，
再在 X-AnyLabeling 里只做「删错的 + 修边缘」：

```python
from ultralytics import SAM

model = SAM("sam2.1_b.pt")
results = model("datasets/fruit/images/raw/durian/durian_0001.jpg",
                points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.9)
for poly in results[0].masks.xy:      # 每个 mask 一个多边形（像素坐标）
    print(len(poly), poly[:3])        # 需要自己归一化后写进 txt
```

注意：everything 模式会把**背景物体也分出来**（砧板、桌子、手），
所以它只适合当"候选池"，必须人工筛。想省这一步，就还是在标注工具里点选式使用 SAM。

---

## 9. 质检（必做，30 秒）

```bash
# 标注完、划分前
python scripts/check_seg_labels.py --images datasets/fruit/images/raw \
    --labels datasets/fruit/labels/raw --nc 2

# 划分后
python scripts/check_seg_labels.py --data configs/fruit-seg.yaml
```

会查出：图片/标签不配对、字段数不对（误用检测格式）、点数不足、
坐标超出 0~1（忘了归一化）、类别 id 越界、**多边形自相交**、面积极小、疑似重复。

输出还有：每类实例数、每图平均实例数、多边形面积分布（判断标注是否过松/过紧）。

---

## 10. 划分 train / val

```bash
# 先看怎么分
python scripts/split_dataset.py --dry-run

# 正式划分（按「图内类别组合」分层，保证 val 每类都有）
python scripts/split_dataset.py --val-ratio 0.2
```

划分后结构变成 `images/{train,val}` + `labels/{train,val}`，
文件名加类别前缀避免不同类别重名。

---

## 11. 训练

```bash
python scripts/04_train_coco8.py --data configs/fruit-seg.yaml \
    --model yolo11n-seg.pt --epochs 50 --name train_fruit_seg
```

或者直接：

```python
from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")          # 必须用 -seg 权重，不是 yolo11n.pt
model.train(data="configs/fruit-seg.yaml", epochs=50, imgsz=1280, batch=4)
```

- `imgsz`：果实在画面里小就用 1024/1280；显存不够降 batch
- 想先验证流程能跑通，就 `epochs=3`，看 `runs/segment/train_fruit_seg/` 里有没有 mask 预览图

指标里 **Mask mAP50-95** 才是分割质量；`Box mAP` 只是框的质量，别混着看。

---

## 12. 常见坑清单

- [ ] 用了 `yolo11n.pt` 而不是 `yolo11n-seg.pt` → 报错或训练成检测
- [ ] 导出成 Detection 格式（5 列）→ `len(segments) != len(boxes)`
- [ ] 忘记归一化（写进 txt 的是像素坐标）→ 坐标 >1，训练时 mask 全飞
- [ ] 类别 id 顺序和 `configs/fruit-seg.yaml` 不一致 → 标签看起来对，类别全错
- [ ] 多边形点序交叉成「8」字 → mask 出现负数面积区域（质检脚本会报）
- [ ] 密集成堆只标了外轮廓 → 模型学成"一坨"，推理时数不清个数
- [ ] val 里某个类别一张都没有 → 该类 mAP 恒为 0，白训
- [ ] 只有几十张图就上 `yolo11x-seg` → 过拟合，n/s 足够
