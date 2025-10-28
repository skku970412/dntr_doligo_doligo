# 미소 객체 검출을 위한 DeNoising FPN 및 Transformer R-CNN

![method](./dnfpn_v2.pdf)


이 저장소는 DNTR(DeNoising Transformer R-CNN)을 PyTorch로 구현하고, 학습된 가중치를 제공합니다. FPN 융합 과정에서 발생하는 노이즈를 줄이기 위해 DN-FPN 플러그인을 도입하고, 표준 R-CNN을 Transformer 기반 구조(Trans R-CNN)로 재구성했습니다.

## 최신 소식
[2024/7/1]: **DQ-DETR** has been accepted by ECCV 2024. 🔥🔥🔥

[2024/5/3]: **DNTR** has been accepted by TGRS 2024. 🔥🔥🔥


## 관련 미소 객체 검출 연구
<!-- A DeNoising FPN With Transformer R-CNN for Tiny Object Detection
Hou-I Liu and Yu-Wen Tseng and Kai-Cheng Chang and Pin-Jyun Wang and Hong-Han Shuai, and Wen-Huang Cheng 
IEEE Transactions on Geoscience and Remote Sensing
[paper] [code]  -->

| Title | Venue | Links | 
|------|-------------|-------|
| **DNTR** | TGRS 2024  | [Paper](https://arxiv.org/abs/2406.05755) \| [code](https://github.com/hoiliu-0801/DNTR) \| [中文解读](https://blog.csdn.net/qq_40734883/article/details/142579516) | 
| **DQ-DETR**| ECCV 2024 | [Paper](https://arxiv.org/abs/2404.03507)  \| [code](https://github.com/hoiliu-0801/DQ-DETR) \| [中文解读](https://blog.csdn.net/csdn_xmj/article/details/142813757) | 


## 설치 및 시작하기

<!-- Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod) -->

본 프로젝트는 [mmdet-aitod](https://github.com/Chasel-Tsui/mmdet-aitod)와 [MMDetection](https://github.com/open-mmlab/mmdetection)을 기반으로 합니다.
<!-- This implementation is based on [MMDetection 2.24.1](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements,  -->

## 환경 구성 안내

`mmdet-dntr/` 디렉터리에는 다음과 같이 두 개의 Python 가상환경을 유지합니다.

| 환경 | 경로 | 용도 |
| --- | --- | --- |
| Runtime | `mmdet-dntr/.venv` | 학습·추론 등 기본 작업(최소 의존성) |
| Test | `mmdet-dntr/testvenv` | 포맷터·테스트 도구·MMTracking 등 부가 의존성 포함 |

잠금 파일을 사용해 아래 스크립트로 손쉽게 생성할 수 있습니다.

```shell script
git clone https://github.com/hoiliu-0801/DNTR.git
cd DNTR/mmdet-dntr
# 두 환경을 모두 생성 (--runtime 또는 --test 로 개별 생성 가능)
./install.sh --all

# 학습/추론 시 Runtime 환경 활성화
source .venv/bin/activate
```

- `configs/_base_/datasets/*.py`의 데이터 경로는 `data/<dataset>/...` 형태로 변경했습니다. 예를 들어 `data/aitod/images/trainval/` 구조에 맞춰 데이터를 배치하세요.
- 각 설정 파일의 `load_from`/`resume_from` 값은 `work_dirs/...`를 기준으로 하므로, 체크포인트를 `mmdet-dntr/work_dirs/` 아래에 두거나 실행 시 경로를 덮어쓰면 됩니다.
- `tools/` 디렉터리의 유틸리티 스크립트는 더 이상 절대경로에 의존하지 않고, 리포지토리 기준 상대경로로 결과를 저장합니다.

## 사용 방법

Runtime 환경을 활성화한 뒤 단일 GPU에서 다음과 같이 실행할 수 있습니다.

학습 예시:

```
python tools/train.py configs/aitod-dntr/aitod_DNTR_mask.py
```

평가 예시:
```
python tools/test.py configs/aitod-dntr/aitod_DNTR_mask.py
```

시각화, PSNR 계산, 데이터셋 변환 등 부가 스크립트를 실행하려면 `testvenv`를 활성화하세요.

```bash
source testvenv/bin/activate
python tools/analysis_tools/analyze_psnr_aitod.py
```

## Performance
표 1. **학습 데이터:** AI-TOD trainval, **평가 데이터:** AI-TOD test, 36 epoch (FRCN은 Faster R-CNN, DR은 DetectoRS).
|Method | Backbone | mAP | AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>vt</sub> | AP<sub>t</sub>  | AP<sub>s</sub>  | AP<sub>m</sub> |
|:---:|:---:|:---:|:---:|:---:|:---:|:---: |:---: |:---: |
FRCN | R-50 | 11.1 | 26.3 | 7.6 | 0.0 | 7.2 | 23.3 | 33.6 | 
ATSS | R-50 | 12.8 | 30.6 | 8.5 | 1.9 | 11.6 | 19.5 | 29.2 | 
ATSS w/ DN-FPN | R-50 | 17.9 | 41.0 | 12.9 | 3.7 | 16.4 | 25.3 | 35.0 |
NWD-RKA | R-50 | 23.4 | 53.5 | 16.8 | 8.7 | 23.8 | 28.5 | 36.0 |
DNTR | R-50 | 26.2 | **56.7** | 20.2 | 12.8 | 26.4 | 31.0 | 37.0 | 
DNTR (New) | R-50 | **27.2** | 56.3 | **21.8** | **15.2** | **27.4** | **31.9** | **38.5** |

표 2. **학습 데이터:** VisDrone train, **검증 데이터:** VisDrone val, 12 epoch 기준.
|Method | Backbone |AP| AP<sub>50</sub> | AP<sub>75</sub> |
|:---:|:---:|:---:|:---:|:---:|
DNTR | R-50 | 34.4 | 57.9 | 35.3 |
UFPMP w/o DN-FPN| R-50 | 36.6 | 62.4 | 36.7 |
UFPMP w/ DN-FPN | R-50 | **37.8** | **62.7** | **38.6** |

## AI-TOD-v1/v2 데이터셋 (⭐️ 부탁드려요!)
* 1단계: 아래 링크에서 데이터셋을 내려받습니다.
```sh
https://drive.google.com/drive/folders/1CowS5BrujefWQxxlmOFfUuLOfUUm8w6U?usp=sharing
```


## AI-TOD-v1/v2 사전 학습 가중치
https://drive.google.com/drive/folders/1i0mYPQ3Cz_k4iAIvSwecwpWMX_wivxzY


## 참고
다른 베이스라인 모델에 DN-FPN을 적용하려면 `mmdet/models/detectors/two_stage_ori.py`를 `mmdet/models/detectors/two_stage.py`로 교체하세요.

예) Faster R-CNN: `python tools/train.py configs/aitod-dntr/aitod_faster_r50_dntr_1x.py`

## 인용
```bibtex
@ARTICLE{10518058,
  author={Liu, Hou-I and Tseng, Yu-Wen and Chang, Kai-Cheng and Wang, Pin-Jyun and Shuai, Hong-Han and Cheng, Wen-Huang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A DeNoising FPN With Transformer R-CNN for Tiny Object Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
}

@InProceedings{huang2024dq,
author={Huang, Yi-Xin and Liu, Hou-I and Shuai, Hong-Han and Cheng, Wen-Huang},
title={DQ-DETR: DETR with Dynamic Query for Tiny Object Detection},
booktitle={European Conference on Computer Vision},
pages={290--305},
year={2025},
organization={Springer}
}
