# Reproducible Environment & Usage Guide (AITODv2 DNTR Evaluation)

This document freezes the working setup that produced the recent 10-/50-image smoke tests. Follow the steps below to reproduce the environment, prepare data, and run evaluations with LRP / area-range metrics enabled.

---

## 1. Python & Virtual Environment

```bash
cd /path/to/DNTR/mmdet-dntr
python3.10 -m venv testvenv
source testvenv/bin/activate
python -m pip install --upgrade pip
```

### 1.1 Base Dependencies

Install the pinned package set (`requirement_test.txt`) that matches the working environment:

```bash
pip install -r requirement_test.txt
```

> **Note**: If you reinstall PyTorch later, reapply the Torch patch in §2.

---

## 2. Torch Import Patch

Some Python builds treat certain C-extension types as immutable, raising:

```
TypeError: cannot set '__module__' attribute of immutable type 'torch._C.DisableTorchFunctionSubclass'
```

Apply the guard once after (re)installing PyTorch:

```bash
python - <<'PY'
from pathlib import Path
path = Path('testvenv/lib/python3.10/site-packages/torch/__init__.py')
old = "if name not in ['DisableTorchFunction', 'Generator']:\n                    obj.__module__ = 'torch'"
new = "if name not in ['DisableTorchFunction', 'Generator']:\n                    try:\n                        obj.__module__ = 'torch'\n                    except TypeError:\n                        pass"
text = path.read_text()
if old not in text:
    raise SystemExit('Torch patch already applied or block missing')
path.write_text(text.replace(old, new))
print('Patched torch/__init__.py')
PY
```

---

## 3. Install AITOD COCO API (LRP Support)

Install the customized COCO API shipped in the repo. Two edits avert duplicate compilation:

```bash
cd ../cocoapi-aitod/aitodpycocotools
# Remove distutils inline sources hint
python - <<'PY'
from pathlib import Path
path = Path('aitodpycocotools/_mask.pyx')
text = path.read_text()
path.write_text(text.replace('# distutils: sources = ../common/maskApi.c', '# distutils: sources ='))
print('Updated _mask.pyx')
PY

# Point setup.py to ../common
python - <<'PY'
from pathlib import Path
path = Path('setup.py')
text = path.read_text()
text = text.replace("os.path.join(here, 'common', 'maskApi.c')",
                    "os.path.join(here, '..', 'common', 'maskApi.c')")
text = text.replace("os.path.join(here, 'common')",
                    "os.path.join(here, '..', 'common')")
path.write_text(text)
print('Updated setup.py')
PY

/home/work/llama_young/dntr_doligo/DNTR/mmdet-dntr/testvenv/bin/pip install --no-build-isolation .
cd ../../mmdet-dntr
```

Verify:

```bash
PYTHONNOUSERSITE=1 testvenv/bin/python -c "import aitodpycocotools; print(aitodpycocotools.__version__)"
```

---

## 4. Dataset Layout

Unzip/download AI-TOD v2 into `data/aitod/` (same layout used by the configs):

```
data/aitod/
├── images/
│   ├── trainval/
│   ├── test/
│   └── ...
└── annotations/
    ├── aitodv2_trainval.json
    └── aitodv2_test.json
```

Ensure the annotation JSONs contain an `"info"` block. You can add a minimal stub via:

```bash
python - <<'PY'
import json
from pathlib import Path
for name in ['aitodv2_trainval.json', 'aitodv2_test.json']:
    path = Path('data/aitod/annotations') / name
    with path.open() as f:
        data = json.load(f)
    data.setdefault('info', {'description': 'AITOD dataset', 'version': 'v2'})
    with path.open('w') as f:
        json.dump(data, f)
    print('Ensured info block in', path)
PY
```

---

## 5. Config & Checkpoint

Use the config reconstructed from `epoch_36.pth` metadata:

```
configs/aitod-dntr/aitod_CL_mask_from_ckpt.py
```

This config already points to `data/aitod/` and matches the class count expected by the checkpoint at `work_dirs/aitod_CL_mask/epoch_36.pth`.

---

## 6. Running Evaluations

### 6.1 Environment guards

To prevent user-site packages (e.g., conflicting PyTorch installs) from shadowing the venv, always run with:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

### 6.2 10-/50-image smoke tests

```bash
# 10 images
PYTHONNOUSERSITE=1 testvenv/bin/python tools/test.py \
  configs/aitod-dntr/aitod_CL_mask_from_ckpt.py \
  work_dirs/aitod_CL_mask/epoch_36.pth \
  --max-samples 10 \
  --eval bbox \
  --work-dir work_dirs/test_subset_max10_ckptcfg

# 50 images (LRP + vt/t/s/m stats)
PYTHONNOUSERSITE=1 testvenv/bin/python tools/test.py \
  configs/aitod-dntr/aitod_CL_mask_from_ckpt.py \
  work_dirs/aitod_CL_mask/epoch_36.pth \
  --max-samples 50 \
  --eval bbox \
  --work-dir work_dirs/test_subset_max50_ckptcfg
```

### 6.3 Full-test evaluation

Simply drop the `--max-samples` flag to evaluate the entire AI-TOD test set:

```bash
PYTHONNOUSERSITE=1 testvenv/bin/python tools/test.py \
  configs/aitod-dntr/aitod_CL_mask_from_ckpt.py \
  work_dirs/aitod_CL_mask/epoch_36.pth \
  --eval bbox \
  --work-dir work_dirs/test_full_ckptcfg
```

All vt/t/s/m and LRP columns will align with the README table when the full dataset is used.

---

## 7. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `TypeError: cannot set '__module__'...` on `import torch` | Re-run the Torch patch in §2 (happens after reinstalling torch). |
| `ModuleNotFoundError: aitodpycocotools` | Reinstall the patched COCO API as in §3, ensuring `_mask.pyx`/`setup.py` edits persist. |
| `KeyError: 'info'` while loading AITOD annotations | Run the JSON patch script in §4 to inject an `info` block. |
| Missing vt/t/s/m or LRP metrics | Confirm `aitodpycocotools` is imported (grep `cocoeval`), and rerun test with `PYTHONNOUSERSITE=1`. |

---

## 8. References

- `report_10.md`, `report_50.md`: smoke-test metrics and console logs.
- `requirement_test.txt`: frozen dependency versions used for these runs.
- `configs/aitod-dntr/aitod_CL_mask_from_ckpt.py`: config reconstructed from the released checkpoint.

With the above steps, the environment, dataset, and evaluation path are fully reproducible.
