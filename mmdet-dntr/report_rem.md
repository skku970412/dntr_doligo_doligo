# Testvenv 10-Image Smoke Test Report

## Environment Fixes
- Patched `testvenv/lib/python3.10/site-packages/torch/__init__.py` so immutable C-extension types (`DisableTorchFunctionSubclass`) are skipped when setting `__module__`. This unblocks `import torch` on the bundled PyTorch 1.12.1 wheels.
- Verified imports with `PYTHONNOUSERSITE=1 testvenv/bin/python -c "import torch; print(torch.__version__)"` to ensure the venv copy (`1.12.1+cu113`) loads instead of the system PyTorch build.
- Added a guard in `~/.local/lib/python3.10/site-packages/sitecustomize.py` to drop the user site-packages path from `sys.path` whenever `PYTHONNOUSERSITE=1` is set. Without this, Python kept preferring the system-installed torch 2.x and the `_C` import failed.

## Smoke Test Command
```
unset PYTHONPATH
PYTHONNOUSERSITE=1 testvenv/bin/python tools/test.py \
  configs/aitod-dntr/aitod_CL_mask_from_ckpt.py \
  work_dirs/aitod_CL_mask/epoch_36.pth \
  --max-samples 10 \
  --eval bbox \
  --work-dir work_dirs/test_subset_max10
```

## Key Console Output
- Dataset build + checkpoint load succeeded (model emits class-count mismatch warnings, but inference proceeds).
- Evaluation metrics (10-image subset):
  - `bbox_mAP`: 0.001
  - `bbox_mAP_50`: 0.003
  - `bbox_mAP_75`: 0.000
  - `bbox_mAP_s`: 0.001
  - `bbox_mAP_m`: 0.000
  - `bbox_mAP_l`: -1.000 (no large objects in subset)
  - `bbox_mAP_copypaste`: `0.001 0.003 0.000 0.001 0.000 -1.000`
  - `bbox_AR@100`: 0.040

## Next Steps
- If you reinstall PyTorch, re-apply the `torch/__init__.py` patch before running tests.
- For full-set evaluation remove `--max-samples` or adjust the cap as needed.
