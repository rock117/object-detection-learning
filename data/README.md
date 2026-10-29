# 测试媒体

- `images/`：放入待检测图片，例如 `demo.jpg`
- `videos/`：放入待检测视频，例如 `demo.mp4`

> 这里只放**临时测试**媒体。要训练自己的类别（如水果检测），请放到
> `datasets/fruit/images/{train,val}` + `labels/{train,val}`，见 `datasets/fruit/README.md`。

图片推理若不指定 `--source`，脚本会使用 Ultralytics 在线示例图 `bus.jpg`。

示例：

```bash
python scripts/01_predict_image.py --source data/images/demo.jpg
python scripts/02_predict_video.py --source data/videos/demo.mp4
```
