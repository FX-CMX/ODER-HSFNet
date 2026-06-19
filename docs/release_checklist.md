# Open-Source Release Checklist

Recommended contents for a paper-support GitHub release:

- Source code required to build, train, validate, and predict with the model.
- Main model YAML: `models/oder_hsfnet.yaml`.
- Ablation YAMLs that correspond to reported experiments.
- Dataset YAML template and dataset preparation instructions.
- Clear mapping between paper names and implementation names.
- Training, validation, and prediction scripts with reproducible default arguments.
- Environment files: `requirements.txt`, `pyproject.toml`.
- License file and upstream attribution.
- Citation information after the paper metadata is finalized.
- Model weights or download links, if publication policy allows sharing them.
- A result table that states dataset, image size, scale, checkpoint, and metrics.

Do not commit:

- Private datasets or data archives.
- `runs/`, `wandb/`, cache files, or generated plots unless they are curated paper figures.
- Local IDE settings, `__pycache__`, `.pt` checkpoints before deciding a release policy.
- Temporary patch/debug scripts from the experimental workspace.
