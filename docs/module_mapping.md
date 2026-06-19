# Module Mapping

This file maps the paper terminology to the implementation names used in this repository.

| Paper name | Public YAML/code name | Legacy/internal name | Location | Notes |
| --- | --- | --- | --- | --- |
| ODER | `ODER` | same | `ultralytics/nn/modules/block.py` | Occlusion-aware Deformable Edge Rectifier. It uses bounded deformable edge resampling, topology-aware sampling routing, and sample-level residual amplitude modulation. |
| HSSF | `HSSF` | `PreDSConv_HyperACE`, `Scheme1_DSConv_HyperACE` | `ultralytics/nn/modules/block.py` | Hypergraph-State Spatial Fusion. Paper-facing alias for the DSConv + VSS/Mamba-compatible + adaptive hypergraph cross-scale fusion block. |
| OCDetect | `OCDetect` | same | `ultralytics/nn/modules/head.py` | Occlusion-Calibrated Detection Head. Adds a class-agnostic objectness calibration branch and modulates class scores before NMS. |
| OCDetect loss | activated by `OCDetect` in YAML filename/model path | `obj_loss` | `ultralytics/utils/loss.py` | Uses foreground assignment targets as soft objectness supervision. |
| Cross-scale distribution path | `FullPAD_Tunnel`, `DownsampleConv` | same | `ultralytics/nn/modules/block.py` | Implementation-level feature routing used around HSSF. This is not named as a separate paper contribution. |

## Naming Policy

For the open-source release, model YAMLs should use the paper names:

- Use `HSSF` instead of `PreDSConv_HyperACE`.
- Keep `ODER` and `OCDetect` unchanged.
- Keep legacy names in code only for old checkpoints and ablation YAMLs.

The main paper model is:

```text
models/oder_hsfnet.yaml
```
