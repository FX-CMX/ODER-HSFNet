# Results

All values are percentages unless otherwise noted. These tables are extracted from the manuscript and should be updated if the paper tables change.

## Dataset Difficulty Verification

| Dataset | Model | mAP50:95 | mAP50 | AP75 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SBD | YOLOv13 | 48.94 | 72.14 | 55.90 | 74.78 | 68.69 |
| SBD | ODER-HSFNet | 50.68 | 73.34 | 59.07 | 75.78 | 69.83 |
| STBD-08 | YOLOv13 | 77.53 | 92.15 | 88.99 | 88.29 | 86.97 |
| STBD-08 | ODER-HSFNet | 78.38 | 92.33 | 89.06 | 88.18 | 87.55 |
| SCB-D3-S | YOLOv13 | 51.39 | 70.07 | 58.77 | 68.54 | 64.98 |
| SCB-D3-S | ODER-HSFNet | 55.64 | 73.13 | 63.56 | 70.57 | 68.12 |
| CrowdHuman | YOLOv13 | 53.85 | 83.96 | 57.82 | 85.70 | 73.62 |
| CrowdHuman | ODER-HSFNet | 54.98 | 84.31 | 59.56 | 86.03 | 73.67 |
| HCCB | YOLOv13 | 57.39 | 76.70 | 67.69 | 73.82 | 72.87 |
| HCCB | ODER-HSFNet | 60.60 | 80.12 | 70.92 | 74.69 | 78.91 |

## Mainstream YOLO-Series Comparison

| Model | HCCB mAP50:95 | HCCB mAP50 | SCB-D3-S mAP50:95 | SCB-D3-S mAP50 | Params (M) | FLOPs (G) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv6s | 58.32 | 78.696 | 55.74 | 73.66 | 16.31 | 43.90 |
| YOLOv8s | 59.77 | 79.27 | 54.68 | 72.64 | 11.14 | 28.66 |
| YOLOv9s | 59.99 | 78.85 | 55.04 | 71.37 | 7.29 | 27.39 |
| YOLOv10s | 59.73 | 78.69 | 53.94 | 71.66 | 8.07 | 24.80 |
| YOLOv11s | 59.95 | 79.69 | 55.26 | 73.40 | 9.43 | 21.56 |
| YOLOv12s | 57.63 | 76.53 | 55.83 | 73.20 | 9.10 | 19.59 |
| YOLOv13s | 57.39 | 76.70 | 51.39 | 70.07 | 9.03 | 21.00 |
| YOLOv26s | 59.15 | 78.64 | 54.18 | 72.34 | 9.39 | 21.26 |
| ODER-HSFNet | 60.60 | 80.12 | 57.36 | 74.65 | 12.74 | 24.87 |

## ODER Ablation Summary

Final ODER configuration:

- insertion: feature layers X1 and X2
- internal components: DEAO + HDSG + Scale
- sampling hypotheses: K = 8
- bounded offsets: tanh constraint enabled

Key reported gains:

| Dataset | Baseline | ODER |
| --- | ---: | ---: |
| HCCB AP_occ | 46.179 | 47.572 |
| HCCB AP75_occ | 35.100 | 36.374 |
| HCCB AP_SF | 41.727 | 43.480 |
| HCCB R_SF | 79.780 | 80.678 |
| SCB-D3-S AP_occ | 36.642 | 38.937 |
| SCB-D3-S AP_SF | 33.567 | 34.785 |

## HSSF Ablation Summary

| Design | HCCB mAP50:95 | HCCB mAP50 | SCB-D3-S mAP50:95 | SCB-D3-S mAP50 | Params (M) | FLOPs (G) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ODER | 58.31 | 77.97 | 52.90 | 71.22 | 9.36 | 23.65 |
| Base flow | 58.94 | 77.90 | 53.27 | 71.95 | 8.79 | 22.90 |
| VSS proxy | 59.67 | 79.34 | 55.25 | 73.43 | 12.18 | 24.22 |
| Hypergraph | 57.99 | 77.12 | 53.72 | 72.74 | 9.32 | 23.53 |
| Local flow | 58.23 | 77.58 | 53.27 | 71.89 | 8.83 | 23.03 |
| HSSF | 60.34 | 79.96 | 55.61 | 73.58 | 12.74 | 24.97 |

## OCDetect Ablation Summary

| Dataset | Model | Head | Pre-NMS | FPPI_pre | mAP50:95 | mAP50 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| HCCB | YOLOv13 | Detect | 1521.59 | 223.40 | 57.39 | 76.70 |
| HCCB | YOLOv13 | OCDetect | 1403.95 | 205.45 | 57.84 | 77.27 |
| HCCB | ODER + HSSF | Detect | 1591.76 | 217.68 | 60.34 | 79.96 |
| HCCB | ODER + HSSF | OCDetect | 1329.76 | 197.55 | 60.60 | 80.12 |
| SCB-D3-S | YOLOv13 | Detect | 270.28 | 41.91 | 51.39 | 70.07 |
| SCB-D3-S | YOLOv13 | OCDetect | 119.09 | 11.04 | 52.79 | 70.45 |
| SCB-D3-S | ODER + HSSF | Detect | 227.77 | 35.01 | 55.61 | 73.58 |
| SCB-D3-S | ODER + HSSF | OCDetect | 117.98 | 10.28 | 57.36 | 74.65 |
