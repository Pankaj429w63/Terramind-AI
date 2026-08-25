# Terramind-AI (MVPDR)

This repository contains a minimal training pipeline for the PlantWild dataset. It includes a safe synthetic trainer and helpers to produce reproducible artifacts on CPU-only Windows hosts.

Quick commands (PowerShell):
```powershell
$env:OMP_NUM_THREADS=1
$env:MKL_NUM_THREADS=1
$env:OPENBLAS_NUM_THREADS=1
$env:NUMEXPR_NUM_THREADS=1
python -u mvpdr/synthetic_train.py
Get-Content outputs/logs/training.log -Tail 50
```

Or build the container:
```bash
docker build -t terramind-ai:latest .
docker run --rm terramind-ai:latest
```

Artifacts produced:
- `outputs/configs_effective.yaml`
- `outputs/logs/training.log`
- `outputs/checkpoints/*`
- `outputs/metrics.json`
- `models/final_model.pth`
 
