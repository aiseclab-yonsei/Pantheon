# Pantheon (Effort) 재현 패키지

기존 Pantheon(Effort)의 Table 5 결과를 재현하고, 새로운 얼굴 이미지에 QCN → CR-FIQA(L) → Effort → Pantheon 추론을 실행하는 독립 작업 폴더입니다.

원본 `/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa`, QCN, CR-FIQA 및 Effort 코드는 수정하지 않았습니다. 필요한 자료를 **복사한 후 이 폴더 안에서만** 실행 코드와 환경을 정리했습니다. 데이터·가중치는 심볼릭 링크가 아닌 실제 파일 복사본입니다.

## 바로 실행

GitHub에는 실행 환경 폴더를 올리지 않습니다. Python 3.11~3.13이 설치된 Linux x86_64에서 `bash setup.sh`를 실행하면 독립 환경을 만들고 필요한 패키지를 설치·검증합니다. NVIDIA GPU가 있으면 지원 런타임에 맞는 PyTorch를 선택합니다. 이 서버에는 데이터·가중치 복사본도 준비되어 있습니다.

```bash
cd Pantheon

# 환경 자동 설치: NVIDIA GPU 감지 시 CPU 재현 환경과 GPU 추론 환경 모두 설치
bash setup.sh

# CPU 데모: FF++로 gate 학습 후 Celeb-DF-v1의 AUC 재현
bash scripts/demo.sh

# Table 5 전체 재현: FF++ 3개 압축 수준 + 교차 데이터셋 10개
bash scripts/reproduce.sh

# Python 진입점을 직접 호출할 때
environment/cpu/bin/python main.py --all-models

# 이미지 8장: 품질 측정부터 Pantheon 예측까지 실제 모델 추론
bash scripts/demo_images.sh

# 원본 파일의 크기·수정 시각 유지 및 결과 일치 확인
environment/cpu/bin/python scripts/verify.py --check-originals
```

전체 재현은 수백만 프레임의 저장 점수를 읽기 때문에 데모보다 오래 걸립니다. 기본 CPU 스레드는 4개이며 `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` 환경 변수로 조절할 수 있습니다.

## 구성

```text
pantheon/
  core.py          원본의 활성 수치 계산 함수, 중복 정의·주석 처리 실험 제거
  pipeline.py      gate 학습, 전체 벤치마크 평가, 표·그림·기준값 비교
  infer.py         이미지에서 QCN, CR-FIQA, Effort, Pantheon 추론
  quality.py       기존 FF++ 기준 품질 정규화
  prepare.py       품질 점수 정규화와 모델 예측 CSV 병합
  score.py         기존/신규 merged CSV에 Pantheon 적용
data/
  merged/          평가에 사용된 11개 merged CSV 복사본
  standardized/    11개 표준화 품질 CSV와 normalization_params.json
  quality_raw/     표준화 이전 CR-FIQA+QCN 품질 CSV 복사본
weights/
  qcn_spaq.pth     QCN SPAQ 학습 가중치
  crfiqa_l.pth     CR-FIQA(L), iresnet100
  effort/          c23, q1~q4 및 비교용 raw, c40, all 가중치
vendor/            필요한 QCN·CR-FIQA 모델 구현과 원 프로젝트 설명
demo/
  input.csv        실물 얼굴 crop 8장의 경로, 정답 및 영상 ID
  images/          복사한 이미지 파일, 원래 영상/프레임 구조 유지
reference/         기존 결과 CSV·그림, CLIP 설정, 복사 출처와 SHA-256
archive/           수정 전 원본 분석·전처리·추론 코드, 참고용
environment/
  cpu/             별도로 설치한 CPU 재현 환경
  gpu/             setup으로 준비한 독립 GPU 추론 환경
  gpu.legacy-*/    예전 복사 환경의 보관본 (Git 제외)
  *.freeze.txt     설치 패키지 버전 기록
outputs/           새 실행 결과와 검증 기록
```

`archive/`는 원본 보존용입니다. 실행 진입점은 `scripts/`와 `pantheon/`입니다. 원본에 남아 있던 절대 경로, 중간 실험 주석, 사용하지 않는 UI 강제 종료와 GPU 번호 강제 지정은 새 실행 경로에서 제거했습니다.

## 재현 기준과 결과

기준 파일은 `reference/ALL_results_frame_video_auc_with_deltas.csv`입니다. `model=ens_adapt`의 `video_auc`가 Pantheon 값이며 단위는 %입니다.

| 평가 데이터셋 | Pantheon video AUC (%) |
|---|---:|
| FF++ raw | 100.00 |
| FF++ c23 | 100.00 |
| FF++ c40 | 99.79 |
| Celeb-DF-v1 | 98.26 |
| Celeb-DF-v2 | 98.15 |
| Celeb-DF-v3 / CDF++ | 93.65 |
| DFDC | 91.32 |
| DFo | 99.54 |
| Wilddeepfake | 89.82 |
| FFIW | 90.87 |
| KoDF | 93.38 |
| DF40 | 82.63 |
| DF-eval | 68.50 |
| 교차 데이터셋 10개 평균 | **90.61** |

FF++ 세 값의 평균은 99.93%, 교차 데이터셋 평균의 반올림 전 값은 90.61374343684233%입니다. Table 5의 다른 논문 방법(DML 등)은 이 패키지의 실행 대상에 포함되지 않습니다.

재현 코드가 생성하는 주요 파일:

- `outputs/reproduction/results.csv`: Pantheon과 Effort 비교 모델의 frame/video AUC
- `outputs/reproduction/pantheon_table5.csv`: Pantheon 13개 평가값
- `outputs/reproduction/summary.json`: 교차 데이터셋 평균
- `outputs/reproduction/table5.png`: 재생성한 결과표
- `outputs/reproduction/reference_comparison.csv`: 기존 결과와의 차이
- `outputs/reproduction/calibration/pantheon.joblib`: 학습된 gate와 RBF 중심

평가 시 프레임 점수를 영상별 평균으로 묶고, 서로 다른 정답이 섞인 영상은 제외합니다. FFIW의 real/fake 영상 구분, Wilddeepfake의 충돌 방지 규칙, DF40의 비영상 생성기 제외 규칙도 원본과 같습니다. 기준값 비교가 허용 오차 `1e-8` percentage point를 넘으면 실행이 실패합니다. 전체 데이터가 포함되지 않은 데모의 평균은 해당 데모 데이터셋의 평균입니다.

## Pantheon 계산 설정

- FF++ 표준화 품질 CSV에서 Q1~Q4의 RBF 중심 계산, sigma=1.0
- FF++의 모든 real 프레임과 c40 fake 프레임으로 gate 학습
- gate 학습에는 c23 점수가 `(0.05, 0.95)` 밖인 고확신 프레임 사용
- gate 입력: `conf, q_std, q_margin, q_range`
- HistGradientBoosting: depth=4, learning_rate=0.05, max_iter=400, seed=0
- 낮은 확신 구간은 RBF soft score 사용
- 나머지는 gate의 오답 확률에 따라 global/soft 혼합 비율 조절
- lambda 범위 0.8→0.2, 오답 확률 경계 0.05와 0.40

정규화 파라미터는 `data/standardized/normalization_params.json`에 보존했습니다. 새 입력에는 이 기준을 그대로 적용합니다.

## 새 이미지 실행

입력은 **이미 추출한 얼굴 crop 이미지**입니다. 원본 영상의 얼굴 검출·정렬 과정은 포함하지 않습니다. 동봉한 데모도 실제 기존 평가 얼굴 crop을 복사한 것입니다.

입력 CSV에는 헤더가 필요하며 `path`는 CSV 파일 위치를 기준으로 한 상대 경로 또는 절대 경로입니다.

```csv
path,label,video_id
images/video_a/000.png,0,video_a
images/video_b/000.png,1,video_b
```

`label`은 0=real, 1=fake입니다. 추론만 할 때는 생략할 수 있습니다. 추론 점수 계산에 입력 정답은 사용하지 않습니다. `video_id`를 제공하면 영상별 평균 점수도 저장합니다.

```bash
bash scripts/demo_images.sh --input /absolute/path/input.csv \
  --output outputs/my_images --device cuda:0

# 비교용 raw/c40/all까지 총 8개 Effort 모델을 실행하려면
bash scripts/demo_images.sh --all-models

# 동일한 추론 코드를 CPU에서 실행할 수도 있습니다. 대형 모델이므로 느립니다.
bash scripts/demo_images.sh --device cpu
```

출력은 `quality_raw.csv`, `merged_predictions.csv`, `frame_predictions.csv`이며 `ens_adapt`가 최종 fake 점수입니다. 기본 판정 경계는 0.5입니다. 영상 ID가 있으면 `video_predictions.csv`도 생성됩니다.

QCN은 SPAQ checkpoint, RGB/Resize(384)/ImageNet 정규화, 원본·좌우반전 특징 평균 및 score pivot 보간을 사용합니다. CR-FIQA는 원본처럼 BGR, 112×112 입력을 사용합니다. Effort는 RGB, bicubic 224×224, CLIP 정규화를 적용합니다. Effort checkpoint의 SVD main/residual 가중치를 합쳐 같은 dense 연산을 구성하므로 별도 CLIP 모델 다운로드가 필요하지 않습니다. 모든 checkpoint는 strict load로 확인합니다.

**저장 CSV를 이용한 표 재현과 이미지 재추론은 구분됩니다.** 이미지 재추론은 CUDA 연산·배치 크기·혼합 정밀도에 따라 기존 저장 점수와 작은 차이가 날 수 있습니다. QCN의 기본 GPU 실행은 원본처럼 float16 autocast를 사용하며 `--no-amp`로 끌 수 있습니다. 데모의 기존 점수는 `reference/demo_stored_scores.csv`에 보존되어 있습니다.

## 품질·예측 CSV부터 병합하기

직접 구한 품질 CSV는 `path,cr_fiqa,qcn` 열을, 예측 CSV는 `path,label,c23,q1,q2,q3,q4` 열을 사용합니다. 품질 CSV의 `rel_path`, `image_path` 열 이름도 허용합니다. raw/c40/all 예측 열은 선택입니다. 경로는 두 파일에서 동일한 키여야 하며, FF++의 첫 데이터셋 디렉터리명에 붙은 `_raw/_c23/_c40`는 원본처럼 정규화합니다. 중복·누락 키와 NaN은 조용히 건너뛰지 않고 오류로 처리합니다.

```bash
environment/cpu/bin/python -m pantheon.prepare \
  --quality outputs/image_demo/quality_raw.csv \
  --predictions outputs/image_demo/merged_predictions.csv \
  --output outputs/prepared/merged_demo.csv

environment/cpu/bin/python -m pantheon.score \
  --input outputs/prepared/merged_demo.csv \
  --output outputs/prepared/pantheon_predictions.csv
```

원본 추론 결과의 모델별 `frame_predictions.csv`에 해당하는 예측값은 이미 `data/merged/`에 모델별 열로 들어 있습니다. 이 폴더는 다른 실험 전체나 원본 데이터셋 전체를 복제하지 않습니다. Table 5 재현에 필요한 CSV 전체와 추론에 필요한 가중치·모델 코드, 데모 이미지를 포함합니다.

## 환경 자동 설치

환경 디렉터리를 배포하거나 Conda 환경을 복사할 필요가 없습니다. Linux x86_64, Python **3.11~3.13**, `venv`가 필요하며, setup은 설치된 호환 Python을 자동으로 찾습니다. 시스템 Python·기존 QCN/CR-FIQA 프로젝트는 수정하지 않고 프로젝트 안의 독립 환경에만 설치합니다.

```bash
# 자동 선택: GPU 사용 가능하면 CPU + GPU, 없으면 CPU 표 재현 환경
bash setup.sh

# 저장 CSV로 표를 재현하는 데 필요한 가벼운 환경만 설치
bash setup.sh cpu

# NVIDIA GPU용 이미지 추론 환경과 CPU 표 재현 환경 설치
bash setup.sh gpu

# NVIDIA GPU 없이 이미지 모델까지 실행하려면 CPU용 PyTorch 설치
bash setup.sh inference-cpu

# 설치 상태만 확인. 패키지 변경이나 다운로드 없음
bash setup.sh --check
```

자동 설치는 다음 작업을 수행합니다.

1. 호환 Python 검색 및 `environment/cpu`, 필요 시 `environment/gpu` 생성
2. `requirements.txt`의 NumPy/Pandas/scikit-learn 등 고정 버전 설치
3. `nvidia-smi`가 보고한 지원 CUDA 버전에 따라 cu130/cu128/cu126 PyTorch wheel 선택
4. QCN·CR-FIQA·Effort에 필요한 패키지 설치
5. `pip check`, 실제 모델 모듈 import, 작은 CPU/GPU 연산으로 설치 검증
6. 설치 성공 표시와 `environment/*.freeze.txt` 기록

기존에 복사해 둔 Conda 환경이 있으면 `environment/gpu.legacy-*`에 보관하고, 충돌 없는 새 venv를 만듭니다. 보관본은 GitHub 업로드에서 제외합니다.

환경이 있고 requirements가 그대로이면 다운로드를 생략하고 검증만 합니다. 설치 도중 실패하면 성공 표시가 남지 않으므로 같은 명령을 다시 실행해서 설치를 이어갈 수 있습니다. `--force`는 패키지 설치 단계를 다시 실행합니다.

GPU 환경은 디렉터리 이름이 `gpu`지만, `inference-cpu` 모드에서는 CPU용 PyTorch가 들어갑니다. 별도 환경 활성화는 필요 없습니다. 데모 스크립트도 환경이 없으면 먼저 설치를 수행합니다.

```bash
# Python 직접 지정
PYTHON=python3.12 bash setup.sh

# CUDA backend 직접 지정
bash setup.sh gpu --torch-backend cu128

# 환경 위치 변경
bash setup.sh --env-root /path/to/pantheon-envs
PANTHEON_ENV_ROOT=/path/to/pantheon-envs bash scripts/demo_images.sh
```

GPU 선택은 드라이버가 보고한 최대 CUDA 지원 버전을 기준으로 합니다. 실제 CUDA 연산이 실패하면 setup도 실패하며 원인을 출력합니다. 호스트의 NVIDIA 드라이버는 운영체제에 설치되어 있어야 합니다. CUDA Toolkit이나 Conda는 별도로 설치할 필요가 없습니다. 더 오래된 드라이버를 사용하는 환경은 드라이버를 갱신하거나 `inference-cpu` 모드를 사용하세요.

패키지 정의는 역할별로 나누었습니다.

- `requirements.txt`: CPU 표 재현의 고정 버전
- `requirements-gpu.txt`: QCN·CR-FIQA·Effort 추론 패키지
- `requirements-torch-cu130.txt`, `requirements-torch-cu128.txt`, `requirements-torch-cu126.txt`: CUDA별 PyTorch 2.9.1 / torchvision 0.24.1
- `requirements-torch-cpu.txt`: CPU용 PyTorch / torchvision

PyTorch wheel은 공식 인덱스에서 먼저 설치하고, 나머지 패키지는 PyPI에서 설치합니다. [PyTorch 공식 설치 조합](https://pytorch.org/get-started/previous-versions/#v291)을 따릅니다. `requirements-gpu.txt`만 단독 설치하기보다 `setup.sh`를 실행하세요.

### GitHub 배포와 데이터·가중치

`.gitignore`에 `environment/cpu/`, `environment/gpu/`, `data/`, `weights/`, `outputs/`를 지정했습니다. 코드·설치 스크립트·requirements·README·데모 이미지·기존 결과표만 GitHub에 올릴 수 있습니다. 현재 로컬 파일은 삭제하지 않습니다.

setup은 **실행 환경을 자동 설치**합니다. 평가 CSV와 학습 가중치는 Python 패키지가 아니므로 별도 파일 배포가 필요합니다. 아직 공개 다운로드 주소를 정하지 않았으므로 임의의 URL에서 다운로드하지 않습니다. 배포한 파일은 다음 구조로 놓으세요.

```text
data/merged/merged_{dataset}.csv        전체 표 재현: DATASETS의 11개 파일
data/standardized/FaceForensics++_standardized.csv
data/standardized/normalization_params.json  이미지 품질 정규화
weights/qcn_spaq.pth
weights/crfiqa_l.pth
weights/effort/c23.pth
weights/effort/q1.pth ~ q4.pth
weights/effort/raw.pth, c40.pth, all.pth  --all-models 선택 시 추가
```

CPU 데모는 FF++와 Celeb-DF-v1 merged CSV, FF++ standardized CSV가 필요합니다. 이미지 데모도 첫 gate 학습을 위해 FF++ CSV 두 개가 필요합니다. 이미 학습한 `pantheon.joblib`가 있으면 `--calibration /path/to/pantheon.joblib`로 전달해 FF++ CSV 없이 추론할 수 있습니다. 이 경우에도 품질 정규화 JSON과 모델 가중치는 필요합니다. 누락된 기본 파일은 setup 검증에서 목록으로 보여 주며, 환경 설치 완료 여부와 구분합니다.

## 복사 출처와 무결성

`reference/copy_manifest.json`은 원본 경로, 복사 대상, 원본 크기·수정 시각, 복사본 SHA-256 및 수정 여부를 기록합니다. 데모 이미지는 `demo/input.csv`의 `source_key`로 원래 평가 프레임과 연결됩니다. 기존 판테온 분석 코드, 품질 추론 코드와 Effort detector는 `archive/`에 보존했습니다.

```bash
environment/cpu/bin/python scripts/verify.py --hashes --check-originals
```

가중치 전체를 해시 검사하므로 시간이 걸릴 수 있습니다. 원본 접근 없이 확인할 때는 `--check-originals`를 생략하세요. 환경 디렉터리 전체는 파일 해시 manifest의 대상이 아니며 설치 패키지 기록을 별도로 보존합니다.

## 이 폴더에서 완료한 검증

- 독립 CPU 환경에서 전체 평가 실행: 영상 AUC 104개가 기존 CSV와 일치, 최대 차이 약 `1.42e-14` percentage point
- Pantheon 13개 값의 최대 차이 0, 교차 데이터셋 평균 90.61374343684233%
- 복사 GPU 환경에서 실제 이미지 8장에 QCN·CR-FIQA·Effort 8개 checkpoint·Pantheon 실행 완료
- 품질 CSV 정규화/병합 → 별도 Pantheon 추론 왕복 결과 일치
- CPU/GPU 환경 모두 `pip check` 통과; GPU 복사본의 사용하지 않는 충돌 패키지는 제거
- 복사 대상 원본 파일의 크기와 수정 시각 유지 확인, 복사본 SHA-256 확인

검증 상세는 `outputs/validation.json`, `outputs/verification.json`, `outputs/integrity_verification.log`와 각 실행 로그에 저장되어 있습니다.

자동 설치 추가 검증: 빈 환경에서 CPU 데모, CUDA 연산, 이미지 8장의 QCN·CR-FIQA·Effort 8개 모델 추론을 확인했습니다. 새 PyTorch 2.9.1 환경의 Pantheon 점수는 이전 환경과 일치했습니다. 결과는 `outputs/setup_validation.json`에 기록했습니다.
