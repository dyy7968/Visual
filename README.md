# YOLOv5 篮球场景目标检测（ball / person）

## 1. 视频来源

| 项目 | 内容 |
|---|---|
| 名称 | NSF Diego Garcia hosts basketball game (989713) |
| 来源 | Wikimedia Commons（公有领域） |
| 链接 | https://commons.wikimedia.org/wiki/File:NSF_Diego_Garcia_hosts_basketball_game_(989713).webm |
| 规格 | 30 s / 1920×1080 / 30 fps / 899 帧 |

## 2. 类别

- `0` = ball（篮球）
- `1` = person（人）

`nc: 2`，与 `names` 长度一致。

## 3. 数据集与标注

每 4 帧取 1 帧共 225 张，剔除末尾 15 张纯黑帧，有效标注 210 张、883 个框。

| 子集 | 图片 | person 框 | ball 框 | 说明 |
|---|---|---|---|---|
| train | 162 | 537 | 52 | 视频前 80% |
| val | 42 | 271 | 23 | 视频后 20% |

划分：按时间顺序 8:2 切分，分界处额外丢弃 6 帧（frame_00692~00712）作缓冲——抽帧步长为 4，相邻帧近乎重复，随机划分会让近重复帧跨集泄漏；train/val 文件名交集为 0。

标注：COCO 预训练 yolov5s 自动预标注（person(0)/sports ball(32) → person(1)/ball(0)），再用 labelImg 人工逐帧修正。

## 4. 训练

```yaml
# dataset/ball_person.yaml
path: ../dataset
train: images/train
val: images/val
nc: 2
names: ['ball', 'person']
```

```bash
python train.py --data dataset/ball_person.yaml --weights yolov5s.pt \
                --epochs 100 --batch-size 8 --imgsz 640 --device 0 --workers 4
```

环境：Windows 11 / Python 3.11.18 / PyTorch 2.11.0+cu128 / RTX 5060 Laptop 8 GB / YOLOv5 v7.0-570-gcb2f17fd（yolov5s，hyp.scratch-low，SGD lr0=0.01，AMP）。

## 5. 验证集指标

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 42 | 294 | 0.992 | 0.962 | 0.990 | 0.854 |
| ball | 42 | 23 | 0.989 | 1.000 | 0.995 | 0.884 |
| person | 42 | 271 | 0.996 | 0.925 | 0.985 | 0.825 |

## 6. 推理与产物

```bash
python detect.py --weights best.pt --source video.mp4 --conf-thres 0.35 --iou-thres 0.45
```

| 文件 | 规格 |
|---|---|
| result.mp4 | 20.0 s / 1280×720 / 4.72 MB（源视频 9–29 s） |
| frame_01/02/03.png | t=2.0 s（ball 0.76 + 4 person）/ 7.2 s（4 person）/ 17.8 s（ball 0.87 + 7 person） |
| best.pt | 13.7 MB |

## 7. 遇到的问题

1. RTX 50 系（sm_120）用旧 CUDA 版 torch 报 `no kernel image`，改 cu128 解决。
2. `plot_images` 报 `numpy.ndarray has no attribute save`（新版 ultralytics 的 `Annotator.im` 是 BGR 数组），改为 `Image.fromarray(im[..., ::-1])`。
3. `detect.py` 不认识 `.webm`，先用 ffmpeg 转 H.264 mp4。
4. `best.pt` 含优化器状态约 54 MB，只保留模型权重后 13.7 MB。
5. ball 跨场景泛化差：本视频球约 82 px 时稳定检出，换球馆后球仅 20–30 px 即基本失效；person 跨场景稳定。
