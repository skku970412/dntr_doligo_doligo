# 50-Image Evaluation Report (AITODv2)

## Environment Updates (in addition to earlier fixes)
- Installed and patched `aitodpycocotools` from the bundled `cocoapi-aitod` repo so that LRP / area-range metrics (vt, t, s, m) are available. `PYTHONNOUSERSITE=1 testvenv/bin/python -c "import aitodpycocotools"` now works.
- Adjusted `aitodpycocotools/_mask.pyx` and `setup.py` to avoid duplicate compilation issues during the local build (removed Cython distutils source directive, pointed setup to `../common`).

## Command
```bash
unset PYTHONPATH
PYTHONNOUSERSITE=1 testvenv/bin/python tools/test.py \
  configs/aitod-dntr/aitod_CL_mask_from_ckpt.py \
  work_dirs/aitod_CL_mask/epoch_36.pth \
  --max-samples 50 \
  --eval bbox \
  --work-dir work_dirs/test_subset_max50_ckptcfg
```

## Metrics (50-image slice)
- `bbox_mAP`: 0.294
- `bbox_mAP_50`: 0.659
- `bbox_mAP_75`: 0.282
- `bbox_mAP_vt`: 0.116
- `bbox_mAP_t`: 0.207
- `bbox_mAP_s`: 0.489
- `bbox_mAP_m`: 0.556
- `bbox_oLRP`: 0.743 / `bbox_oLRP_Localisation`: 0.297 / `bbox_oLRP_false_positive`: 0.230 / `bbox_oLRP_false_negative`: 0.346
- `bbox_mAP_copypaste`: `0.294 -1.000 0.659 0.282 0.116 0.207`

## Notes
- With the AITOD-specific COCOeval installed, vt/t/s/m splits and LRP stats populate correctly even on partial subsets.
- Per-class metrics still show `nan` when a category is absent in the sampled images. Results should converge toward the README table when evaluating the full test set.
