# YOLOv5 篮球场景目标检测（ball / person）

## 1. 视频来源

| 项目 | 内容 |
|---|---|
| 名称 | NSF Diego Garcia hosts basketball game (989713) |
| 来源 | Wikimedia Commons（公有领域，美国政府作品） |
| 链接 | https://commons.wikimedia.org/wiki/File:NSF_Diego_Garcia_hosts_basketball_game_(989713).webm |
| 规格 | 30.0 s / 1920×1080 / 30 fps / 899 帧 |

## 2. 类别

```
nc: 2
names: ['ball', 'person']
```

- `0` = ball（篮球）
- `1` = person（人）

类别序号与标注文件、配置文件完全一致。

## 3. 数据集与标注

| 项目 | 数值 |
|---|---|
| 抽帧 | 每 4 帧取 1 帧，共 225 张 |
| 有效标注 | 210 张（末尾 15 帧为纯黑渐出，无目标，剔除） |
| 标注框总数 | 883 |

| 子集 | 图片 | person 框 | ball 框 | 说明 |
|---|---|---|---|---|
| train | 162 | 537 | 52 | 视频前 80% |
| val | 42 | 271 | 23 | 视频后 20% |
| 缓冲丢弃 | 6 | — | — | 分界处 frame_00692 ~ frame_00712 |

**划分方式**：按时间顺序 8:2 切分，分界处额外丢弃 6 帧。抽帧步长为 4，相邻帧近乎重复，随机划分会造成近重复帧跨集泄漏，故采用时序切分。train/val 文件名交集为 0。

**标注方式**：先用 COCO 预训练 yolov5s 自动预标注，把 `person(0)`/`sports ball(32)` 映射为 `person(1)`/`ball(0)`，再在 labelImg 中人工逐帧修正。标签为 YOLO 格式 `类别 中心x 中心y 宽 高`（归一化）。

## 4. 训练配置

```yaml
# dataset/ball_person.yaml
path: ../dataset
train: images/train
val: images/val
nc: 2
names: ['ball', 'person']
```

| 组件 | 版本 |
|---|---|
| 系统 / Python | Windows 11 / 3.11.18 |
| PyTorch / CUDA | 2.11.0+cu128 / 12.8 |
| GPU | RTX 5060 Laptop 8 GB（sm_120） |
| YOLOv5 | v7.0-570-gcb2f17fd，模型 yolov5s |

```bash
python train.py --data dataset/ball_person.yaml --weights yolov5s.pt \
                --epochs 100 --batch-size 8 --imgsz 640 \
                --device 0 --workers 4 --patience 30
```

超参 `data/hyps/hyp.scratch-low.yaml`，SGD（lr0=0.01），开启 AMP。

## 5. 验证集指标

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 42 | 294 | 0.992 | 0.962 | 0.990 | 0.854 |
| ball | 42 | 23 | 0.989 | 1.000 | 0.995 | 0.884 |
| person | 42 | 271 | 0.996 | 0.925 | 0.985 | 0.825 |

## 6. 推理与产物

```bash
python detect.py --weights best.pt --source video.mp4 \
                 --conf-thres 0.35 --iou-thres 0.45
```

| 文件 | 规格 |
|---|---|
| `result.mp4` | 20.0 s / 1280×720 / 4.95 MB，源视频第 9–29 秒 |
| `frame_01.png` | t=2.8 s，ball 0.78 + 5 person |
| `frame_02.png` | t=7.2 s，4 person（球在手中未单独框出） |
| `frame_03.png` | t=17.8 s，ball 0.87 + 7 person |
| `best.pt` | 14.4 MB |

## 7. 遇到的问题

| 问题 | 处理 |
|---|---|
| RTX 50 系（sm_120）用旧 cu118/cu121 报 `no kernel image` | 换 torch 2.11.0+cu128 |
| `plot_images` 报 `'numpy.ndarray' object has no attribute 'save'`（新版 ultralytics 的 `Annotator.im` 为 BGR 数组） | 改为 `Image.fromarray(im[..., ::-1])` 再存 |
| `detect.py` 不识别 `.webm` | 先用 ffmpeg 转 H.264 mp4 |
| 视频末尾 0.5 s 纯黑渐出 | 剔除 15 帧无标注黑帧 |
| `best.pt` 含优化器状态约 54 MB，超过单文件限制 | 只保留模型权重，14.4 MB |
| 跨场景泛化：换球馆后 ball 失效 | 见下 |

**ball 类的能力边界（实测）**：视频中球约占画面宽 4%（82 px）时可稳定检出；换到其他球馆或转播画面后球仅 20–30 px 且运动模糊，检出基本失效。person 类跨场景稳定。补充 95 个近景球框重训后，近景球检测可达约 50% 精确率，但该版本验证集 mAP50-95 降至 0.776，故未采用。

## 8. 复现步骤

1. 下载源视频，ffmpeg 转 mp4；
2. 每 4 帧抽 1 帧，共 225 张，剔除末尾黑帧剩 210 张；
3. yolov5s 预标注后人工修正，类别映射为 ball=0 / person=1；
4. 按时间顺序 8:2 划分（分界丢弃 6 帧）→ train 162 / val 42；
5. 写好 `ball_person.yaml`（`nc` 与 `names` 长度一致）；
6. 按第 4 节命令训练 100 轮取 `best.pt`，按第 6 节推理。

数据集未提交，按上述步骤可从源视频重建完全一致的 train/val 划分。
