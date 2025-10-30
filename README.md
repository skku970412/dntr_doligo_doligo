# dntr_doligo_doligo

`dntr_doligo_doligo`는 DNTR(DeNoising Transformer R-CNN)을 기반으로 하는 미소 객체 검출 실험용 저장소입니다. 원 저자의 [공개 구현](https://github.com/hoiliu-0801/DNTR)을 바탕으로 Smart Factory 과제에 맞춰 데이터 경로, 실행 스크립트, 환경 세팅을 정리했습니다.

![method](./dnfpn_v2.pdf)

## 프로젝트 개요
- DN-FPN 플러그인을 이용해 FPN 융합 단계의 노이즈를 줄이고, Transformer 기반 두 단계 검출기를 구성합니다.
- [MMDetection](https://github.com/open-mmlab/mmdetection) 2.x 코드베이스와 [mmdet-aitod](https://github.com/Chasel-Tsui/mmdet-aitod)를 토대로 실험 설정을 재구성했습니다.
- AI-TOD / VisDrone 등 미소 객체 데이터셋 실험을 위해 데이터 구조와 체크포인트 경로를 맞춰 두었습니다.

## 리포지토리 구성
- `mmdet-dntr/` – DNTR 핵심 코드와 설정, 훈련 스크립트
- `cocoapi-aitod/` – AI-TOD 전용 COCO API 확장
- `figures/`, `dnfpn_v2.pdf` – 모델 아키텍처 자료
- `requirements.txt` – 재현을 위한 Python 패키지 목록

## 빠른 시작
```bash
# 1. Clone
git clone https://github.com/skku970412/dntr_doligo_doligo.git
cd dntr_doligo_doligo

# 2. 가상환경 생성 (예: Python 3.10 이상)
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. CUDA/OS에 맞춰 PyTorch 설치 (예시: CUDA 11.8)
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu118

# 4. 공통 의존성 설치
pip install -r requirements.txt

# 5. MMDetection 커스텀 모듈 설치
cd mmdet-dntr
pip install -e .
```

> mmcv-full 설치가 실패하면 `mim install mmcv-full==1.6.0` 명령을 사용해 CUDA에 맞는 바이너리를 설치한 뒤 다시 시도하세요.

## 가상환경 스크립트
`mmdet-dntr/install.sh`는 잠금 파일(`requirements/runtime-lock.txt`, `requirements/test-lock.txt`)을 이용해 두 종류의 가상환경을 생성합니다.

```bash
cd mmdet-dntr
./install.sh --all      # .venv, testvenv 동시 생성
./install.sh --runtime  # Runtime 환경만 갱신
./install.sh --test     # Test 환경만 갱신
```

- Runtime: 학습/추론용 최소 패키지 구성 (`mmdet-dntr/.venv`)
- Test: 포맷터, 테스트 툴, MMTracking 등을 포함한 확장 구성 (`mmdet-dntr/testvenv`)

## 데이터 준비
데이터는 저장소에 포함되지 않습니다. 아래 구조를 지켜 `mmdet-dntr/data/`에 배치하세요.

```
mmdet-dntr/
└── data/
    └── aitod/
        ├── annotations/
        │   ├── trainval.json
        │   ├── test.json
        │   └── ...
        └── images/
            ├── trainval/
            ├── val/
            └── test/
```

- AI-TOD v1/v2: [Google Drive](https://drive.google.com/drive/folders/1CowS5BrujefWQxxlmOFfUuLOfUUm8w6U?usp=sharing)
- VisDrone: 공식 홈페이지에서 다운로드 후 `data/visdrone/` 구조에 맞춰 정리
- 다른 데이터 경로를 사용하려면 `configs/_base_/datasets/*.py` 또는 실행 시 `--cfg-options`로 경로를 수정하세요.

## 학습 & 평가
Runtime 환경을 활성화한 뒤 명령어를 실행합니다.

```bash
# 학습
python tools/train.py configs/aitod-dntr/aitod_DNTR_mask.py

# 평가
python tools/test.py configs/aitod-dntr/aitod_DNTR_mask.py \
    work_dirs/aitod_DNTR_mask/latest.pth

# 추가 스크립트 (예: PSNR 분석) – 필요 시 testvenv 사용
source testvenv/bin/activate
python tools/analysis_tools/analyze_psnr_aitod.py
```

- 결과 로그와 체크포인트는 `mmdet-dntr/work_dirs/` 아래에 생성됩니다.
- 실험 기록은 `work_dirs/logs/`에서 확인할 수 있으며, README에 명시된 구조를 유지하면 바로 재현 가능합니다.

### 스모크 테스트 스크립트

소규모 서브셋(10·50장 등)으로 빠르게 확인하려면 `run_smoke_tests.sh` 를 사용할 수 있습니다.

```bash
cd mmdet-dntr
./run_smoke_tests.sh        # 기본 10장 평가
./run_smoke_tests.sh 50     # 50장 평가 등 원하는 샘플 수 지정
```

`testvenv` 가 설치되어 있어야 하며, 결과는 `work_dirs/test_subset_max{N}_ckptcfg` 폴더에 저장됩니다.

## 사전 학습 가중치
- AI-TOD v1/v2: [Google Drive](https://drive.google.com/drive/folders/1i0mYPQ3Cz_k4iAIvSwecwpWMX_wivxzY)
- 가중치를 `mmdet-dntr/work_dirs/<exp_name>/`에 배치하고 구성 파일의 `load_from` 값을 조정하세요.

## 참고 & 인용
DNTR 및 연관 연구를 인용할 때 아래 BibTeX을 사용하세요.

```bibtex
@article{liu2024dntr,
  author    = {Hou-I Liu and Yu-Wen Tseng and Kai-Cheng Chang and Pin-Jyun Wang and Hong-Han Shuai and Wen-Huang Cheng},
  title     = {A DeNoising FPN With Transformer R-CNN for Tiny Object Detection},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  year      = {2024}
}

@inproceedings{huang2024dq,
  author    = {Yi-Xin Huang and Hou-I Liu and Hong-Han Shuai and Wen-Huang Cheng},
  title     = {DQ-DETR: DETR with Dynamic Query for Tiny Object Detection},
  booktitle = {European Conference on Computer Vision},
  year      = {2025}
}
```

## 라이선스
본 저장소는 Apache License 2.0을 따릅니다. 자세한 내용은 `LICENSE` 파일을 참고하세요.
