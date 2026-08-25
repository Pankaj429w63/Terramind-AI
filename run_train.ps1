# PowerShell helper to run full training with safe thread limits
$env:OMP_NUM_THREADS=1
$env:MKL_NUM_THREADS=1
$env:OPENBLAS_NUM_THREADS=1
$env:NUMEXPR_NUM_THREADS=1
python -u mvpdr\train_mvpdr.py
if ($LASTEXITCODE -ne 0) { Write-Error "Training failed with exit code $LASTEXITCODE" }
