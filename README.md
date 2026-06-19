# ODER-HSFNet

Official code release candidate for the paper **A Smart Classroom Behavior Analysis Framework with a New Highly Congested Classroom Dataset**.

ODER-HSFNet is a YOLO-based detector for highly congested classroom behavior detection. It is designed for dense instance co-occurrence, asymmetric occlusion, depth-wise scale discontinuity, and fine-grained semantic degradation in distant student targets.

The released model is defined in [models/oder_hsfnet.yaml](models/oder_hsfnet.yaml). The three paper-facing components are:

- **ODER**: Occlusion-aware Deformable Edge Rectifier.
- **HSSF**: Hypergraph-State Spatial Fusion.
- **OCDetect**: Occlusion-Calibrated Detection Head.

This repository is organized as a clean research artifact: source code, model YAMLs, runnable scripts, and documentation are included; datasets, training runs, cache files, and private experiment outputs are intentionally excluded.

## Repository Layout

```text
ODER-HSFNet/
├── models/
│   ├── oder_hsfnet.yaml                 # main paper model: ODER + HSSF + OCDetect
│   ├── ablation_oder_hssf_detect.yaml   # ODER + HSSF + standard Detect
│   └── ablation_yolov13_ocdetect.yaml   # YOLOv13 baseline + OCDetect
├── configs/
│   └── dataset_example.yaml             # dataset config template
├── scripts/
│   ├── train.py                         # training entry
│   ├── val.py                           # validation entry
│   └── predict.py                       # inference entry
├── docs/
│   ├── module_mapping.md                # paper names vs implementation names
│   ├── paper_summary.md                 # extracted paper-facing summary
│   ├── dataset_hccb.md                  # HCCB dataset facts and label protocol
│   ├── reproducibility.md               # environment, hyperparameters, metrics
│   ├── results.md                       # paper result tables in compact form
│   └── release_checklist.md             # what should be included before publishing
├── ultralytics/                         # modified YOLOv13/Ultralytics source
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
conda create -n oder-hsfnet python=3.11 -y
conda activate oder-hsfnet
pip install -r requirements.txt
pip install -e .
```

Optional: install `mamba_ssm` if you want the CUDA Mamba branch in HSSF. Without it, the code uses a local convolutional fallback so model construction and training still work.

## Dataset

Create a YOLO-format dataset YAML, for example:

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test

nc: 7
names:
  0: reading
  1: writing
  2: heads up
  3: sleeping
  4: looking around
  5: bowing head
  6: using phone
```

A template is provided at [configs/dataset_example.yaml](configs/dataset_example.yaml).

## Training

```bash
python scripts/train.py \
  --data configs/dataset_example.yaml \
  --model models/oder_hsfnet.yaml \
  --scale s \
  --imgsz 640 \
  --epochs 250 \
  --batch 32 \
  --device 0
```

For DDP:

```bash
python scripts/train.py \
  --data configs/dataset_example.yaml \
  --model models/oder_hsfnet.yaml \
  --scale s \
  --imgsz 640 \
  --epochs 250 \
  --batch 32 \
  --workers 2 \
  --device 0,1
```

The paper setting uses image size 640, 250 epochs, batch size 32, learning rate 0.01, linear decay, optimizer `auto`, momentum 0.937, weight decay 0.0005, NMS IoU 0.7, and close mosaic over the last 10 epochs. If a 2-GPU DDP run is killed by memory pressure, reduce `--batch` to 16 or 8.

## Paper Results

Main reported results:

| Dataset | mAP50:95 | mAP50 | AP75 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| HCCB | 60.60 | 80.12 | 70.92 | 74.69 | 78.91 |
| SCB-D3-S | 57.36 | 74.65 | - | - | - |

See [docs/results.md](docs/results.md) for benchmark, mainstream YOLO comparison, and ablation summaries.

## Validation

```bash
python scripts/val.py \
  --weights runs/ODER-HSFNet/oder_hsfnet_s/weights/best.pt \
  --data configs/dataset_example.yaml \
  --imgsz 640 \
  --device 0
```

## Prediction

```bash
python scripts/predict.py \
  --weights runs/ODER-HSFNet/oder_hsfnet_s/weights/best.pt \
  --source path/to/images_or_video \
  --imgsz 640 \
  --device 0
```

## Notes For Paper Reproducibility

- The main YAML uses the paper name `HSSF`. Older experiment YAMLs used `PreDSConv_HyperACE`; that name is kept as a compatibility alias only.
- `FullPAD_Tunnel` and `DownsampleConv` are implementation-level feature distribution blocks inherited from the YOLO pipeline. They are documented in [docs/module_mapping.md](docs/module_mapping.md).
- The default training scripts do not include private datasets, pretrained checkpoints, or generated experiment folders.

## License

This project is based on YOLOv13/Ultralytics code and keeps the upstream AGPL-3.0 license file. Check dependency and dataset licenses before redistribution.
