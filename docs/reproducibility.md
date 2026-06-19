# Reproducibility

This document mirrors the implementation details reported in the paper.

## Environment

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04 Linux |
| GPUs | 2 x NVIDIA RTX 3090, 24 GB each |
| Python | 3.11.14 |
| PyTorch | 2.2.2 |
| CUDA | 12.1 |
| cuDNN | 8.9.7 |
| OpenCV | 4.9.0 |
| Ultralytics | 8.3.63 |

## Training Hyperparameters

| Hyperparameter | Value |
| --- | --- |
| Learning rate | 0.01 |
| Decay strategy | Linear decay |
| Optimizer | Auto |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Total epochs | 250 |
| Batch size | 32 |
| NMS threshold | 0.7 |
| Image size | 640 |
| Close mosaic epochs | 10 |

## Metrics

Main metrics:

- mAP50:95
- mAP50
- AP75
- Precision
- Recall

Candidate-box and dense-occlusion analysis metrics:

- Pre-NMS: average candidate boxes before non-maximum suppression.
- FPPI_pre: false positive candidate boxes per image before NMS.
- FPPI: false positives per image after standard post-processing.
- FP_nei and FPPI_nei: false positives caused by neighboring instances or occlusion boundaries.
- AP_occ and AP75_occ: occluded-target detection metrics.
- AP_SF and R_SF: distant small-target metrics.
- Params and FLOPs: model complexity.

## Example Training Command

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

If the run is killed by host or GPU memory pressure, use `--batch 16` or `--batch 8`.
