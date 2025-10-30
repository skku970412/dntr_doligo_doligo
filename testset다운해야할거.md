# ✅ 앞으로 진행해야 할 작업

1. **데이터셋 다운로드 & 배치**
   - README에 안내된 링크에서 AI-TOD / VisDrone 등 필요한 데이터셋을 직접 다운로드합니다.
   - `DNTR/mmdet-dntr/data/` 아래에 다음과 같은 구조로 복사해 주세요.
     ```
     mmdet-dntr/data/aitod/images/trainval/
     mmdet-dntr/data/aitod/annotations/...
     mmdet-dntr/data/visdrone/VisDrone2019-DET-train/images/
     ...
     ```

2. **체크포인트 정리 (선택 사항)**
   - 사전 학습 가중치가 있다면 `mmdet-dntr/work_dirs/<exp_name>/` 아래에 두고,
     config의 `load_from` 경로(`work_dirs/...`)와 맞춰 주세요.

3. **환경 활성화 및 실행**
   - Runtime 환경 활성화:
     ```bash
     cd DNTR/mmdet-dntr
     source .venv/bin/activate
     ```
   - 예시 실행:
     ```bash
     python tools/test.py configs/aitod-dntr/aitod_DNTR_mask.py work_dirs/aitod_DNTR_mask/latest.pth
     ```
   - 테스트/툴 활용 시:
     ```bash
     source testvenv/bin/activate
     ```

4. **필요 시 config 수정**
   - 데이터 구조가 README와 다르면 `configs/_base_/datasets/*.py` 또는 실행 시 `--cfg-options`로 경로를 조정하세요.

5. **결과 확인 및 추가 검증**
   - 로그/결과는 `mmdet-dntr/work_dirs/`에 저장됩니다.
   - 학습/검증 후 필요한 그래프나 로그를 정리해 주세요.
