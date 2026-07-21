# semantic_nav — OmniVLA-edge 언어 파인튜닝

이 레포는 [NHirose/OmniVLA_edge](https://github.com/NHirose/OmniVLA_edge)(정리 전 연구 코드)를 포크해서,
**커스텀 LeLaN 형식 데이터로 OmniVLA-edge 모델을 언어(instruction) 조건으로 파인튜닝**할 수 있도록
전용 경로(`lan_only_ft`)를 추가한 것입니다.

> ⚠️ 이 문서는 코드 분석에 기반한 정리입니다. 아래 "언어 전용 파인튜닝" 경로는 구조·구문은 검증했지만
> **데이터가 있는 GPU 머신에서 end-to-end로는 아직 실행/검증되지 않았습니다.** 첫 실행 시 사소한 수정이
>필요할 수 있습니다.

---

## 브랜치 안내 (역할별)

목적별로 브랜치가 나뉘어 있습니다. 분기 관계: `lang-direction-ft` ← `episode-split-lang-eval` ← `main`.

| 브랜치 | 역할 | 핵심 내용 |
|---|---|---|
| **`main`** | **안정 파인튜닝 (기본 진입점)** | 언어전용(`use_image_goal: False`) + base-driving cap, **index-split(90/10, 데이터 최대 활용)**, eval 도구 정리(`train/eval/`), mws 0.125 통일, 프롬프트 blocklist(표면/구역 제외), 75/25 language/position mask. 일반 학습·사용은 여기서. |
| **`feature/episode-split-lang-eval`** | **엄밀 평가용** | **에피소드 단위 hold-out**(`split_by_episode` + `test_episodes`)으로 train/test leakage 없는 일반화 평가 + 언어 진단 도구(`prompt_sensitivity.py`=언어 반응성, `grounding_test.py`=객체 선택 정확도, `direction_test.py`=방향 이해 대조검사). |
| **`feature/lang-direction-ft`** | **실험: 방향·언어지시 학습** | episode-split 기반 위에, **방향 명령(left/right/straight) + OOD 표현**을 대조 counterfactual로 학습(`train_lan_aug_ft`, `vint_train/data/lang_aug.py`, `config/frodo_lan_ft_lang.yaml`). 결과: held-out에서 방향 delta +0.54, OOD 표현 +0.50, 객체 grounding 유지. ⚠️ 아직 **scene-free**(장애물 미고려) — 다음 목표는 "장애물 피하며 방향 따르기". 현재 연구 프론티어. |

> 원격(`semantic_nav`)에는 `main`, `feature/episode-split-lang-eval`이 올라가 있고, `feature/lang-direction-ft`는 현재 로컬 작업 중입니다.

---

## 0. 두 개의 서로 다른 OmniVLA 코드베이스 (중요)

혼동을 막기 위해 먼저 구분합니다.

| | **이 레포 (OmniVLA-edge)** | 공개 [NHirose/OmniVLA](https://github.com/NHirose/OmniVLA) |
|---|---|---|
| 모델 | EfficientNet+Transformer, `IL_gps_map_mask3_lan2` (경량) | OpenVLA 기반 (대형) |
| 학습 스크립트 | `train/train.py` | `vla-scripts/train_omnivla.py` (torchrun, 8×H100) |
| LeLaN 궤적 | **학습 중 런타임 생성** (`supervision_from_nomad`) | 데이터셋에 **미리 포함**(`LeLaN_dataset_NoMaD_traj`) |
| 체크포인트 | `omnivla-edge.pth` (아래 참조) | `omnivla-original.pth` 등 |

- `omnivla-edge.pth`는 공개 레포 HF(`huggingface.co/NHirose/omnivla-edge`)에서 받으며,
  **이 레포의 `IL_gps_map_mask3_lan2`와 구조가 100% 동일**합니다(서브모듈 12개, 하이퍼파라미터, forward 시그니처, 반환 모두 일치. `obs_encoding_size=1024`, action head 출력 `32 = 8×4`).
- 용량이 커서 이 레포에는 **커밋하지 않습니다**(`.gitignore`의 `*.pth`). `train/omnivla-edge.pth`로 별도 배치하세요.

---

## 1. 우리가 추가한 것 — 언어 전용 파인튜닝 경로 (`lan_only_ft`)

원본 학습 루프는 매 스텝 **frodobot+GNM+LeLaN(+BDD) 4개 데이터 스트림**을 동시에 사용하고,
LeLaN의 행동 라벨을 **teacher 모델(ExAug/MBRA, NoMaD)로 런타임 생성**합니다. 즉 원본 그대로는
그 데이터들과 teacher 체크포인트가 전부 있어야 돕니다.

우리 목표는 "**내 LeLaN 데이터만으로, teacher 없이**" edge를 파인튜닝하는 것이라, 다음을 확인/구현했습니다.

- 우리 데이터 pickle에는 이미 `nomad_traj_norm (8,4)` 궤적 라벨이 들어 있음 → **런타임 NoMaD 불필요**.
- 모델 출력이 `(8,4)`라 이 궤적을 **그대로 행동 라벨로** 사용 가능.
- 그래서 `lan_only_ft: True`면 **LeLaN 로더 하나만** 띄우고, frodobot/GNM/BDD 데이터와 ExAug/NoMaD teacher를 **전혀 만들지 않습니다.**

### 변경 파일
| 파일 | 내용 |
|---|---|
| `train/config/frodo_lan_ft.yaml` (신규) | 파인튜닝 config. `lan_only_ft: True`, `omnivla-edge.pth` 직접 로드 |
| `train/vint_train/data/data_config.yaml` | `frodo_lan: metric_waypoint_spacing 0.12` 등록 |
| `train/vint_train/data/lelan_dataset.py` | `frodo_lan` 데이터 로딩(`_load_split_index` 분기, `_getitem_frodo_lan`). pickle의 `bbox(1,4)/pose_median(1,2)/pose_median_norm/nomad_traj_norm(8,4)/prompt`를 읽고, `nomad_traj_norm`을 11번째 반환값으로 추가 |
| `train/vint_train/training/train_utils.py` | `map_cache` import 가드 + `train_lan_only_ft`(teacher 없이 `nomad_traj_norm`을 라벨로 학습) |
| `train/train.py` | `diffusion_policy` import 가드 + `main_lan_only_ft` + `main()`/`__main__` 분기 |

---

## 2. 데이터 준비

데이터는 **에피소드별 폴더**로 구성됩니다. 데이터 루트(`<DATA_ROOT>`, 예: `omnivla_dataset/`) 아래에
`episode_XXXX/` 폴더들이 있고, 각 에피소드 안에 `image/`와 `pickle_nomad/`가 있습니다.

```
<DATA_ROOT>/                              # 예: omnivla_dataset/
├── episode_0020/
│   ├── image/00000000.jpg, 00000001.jpg, …      # 224×224 RGB, 에피소드 내 연속(시간순)
│   └── pickle_nomad/00000000.pkl, …             # image와 1:1, 같은 stem
├── episode_0021/
│   ├── image/ …
│   └── pickle_nomad/ …
└── episode_0037/ …
```

**config 키 ↔ 경로 매핑** (`config/frodo_lan_ft.yaml`의 `datasets_lan.frodo_lan`):

| config 키 | 가리켜야 할 것 | 내용 |
|---|---|---|
| `image`  | `<DATA_ROOT>/` (루트)           | 로더가 `episode_*/image/`를 자동 탐색 |
| `pickle` | `<DATA_ROOT>/` (루트, 같은 값)  | 로더가 `episode_*/pickle_nomad/`를 자동 탐색 |
| `train`  | 아무 빈 쓰기가능 폴더           | train LMDB 캐시 생성 위치 |
| `test`   | 아무 빈 쓰기가능 폴더 (train과 다르게) | test LMDB 캐시 생성 위치 |

> `image`와 `pickle`은 **둘 다 데이터 루트(같은 경로)** 를 가리킵니다 — 로더가 그 아래 `episode_*/image`와
> `episode_*/pickle_nomad`를 알아서 찾습니다. 모두 **절대경로 권장**. `train`/`test`는 그냥 캐시용 빈 폴더
> (예: `<DATA_ROOT>/_cache/{train,test}/`, 자동 생성해두면 됨).

**규칙**
- 이미지는 **반드시 224×224**. 로더가 리사이즈하지 않고 bbox/crop이 224 기준(코드에서 안전하게 224로 강제하긴 함).
- 각 에피소드 프레임은 **에피소드 내 연속·시간순**. 로더가 **에피소드 경계를 인식**해 context(`iv-1..iv-5`)를
  같은 에피소드로 클램프합니다.
- 객체 없는 프레임도 pkl(빈 리스트) 저장 OK — 로더가 자동으로 건너뜁니다.
- **train/test 분할은 로더가 전체 프레임의 90/10으로 자동** 수행(당신이 나눌 필요 없음).

**pickle 형식** (프레임당 객체 리스트, 각 원소 dict — 우리 데이터 실측 기준):
```python
{
  "bbox":             np.ndarray (1,4) int,    # [[top, bottom, left, right]], 224 공간
  "pose_median":      np.ndarray (1,2) float,  # [[forward, left]] meters
  "pose_median_norm": np.ndarray (1,2) float,  # pose_median / 0.12
  "nomad_traj_norm":  np.ndarray (8,4) float32,# 누적 (x=fwd, y=left, cos, sin), 0.12 정규화 ← 궤적 라벨
  "prompt":           np.ndarray (N,1) / list, # 지시문들 (로더가 문자열로 언랩)
}
```

---

## 3. 실행 방법

### 3-1. 환경 (conda)

`lan_only_ft` 전용 최소 환경 파일 `environment_frodo_lan.yml`을 제공합니다.

**(0) conda가 없다면 먼저 설치** (이미 `conda` 명령이 되면 건너뛰기 — `conda --version`으로 확인)
```bash
# Miniconda 설치 (Linux x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
# 현재 셸에서 conda 활성화
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash        # 이후 새 터미널부터 conda 자동 사용 (zsh면 conda init zsh)
# 새 터미널을 열거나:  exec bash
```

**(1) 환경 생성** (레포 루트 = 이 README가 있는 폴더에서)
```bash
cd /home/shy/suhyeon/OmniVLA_edge          # 레포 루트로 이동
conda env create -f environment_frodo_lan.yml
```
> 처음엔 몇 분 걸립니다(패키지 다운로드). 다시 만들 땐 `conda env create -f environment_frodo_lan.yml --force`.

**(2) 환경 활성화** — 학습/추론 전에 매번 필요
```bash
conda activate frodo_lan
```
프롬프트 앞이 `(base)` → `(frodo_lan)`으로 바뀌면 성공.

**(3) 설치 확인 (GPU 인식되는지)**
```bash
python -c "import torch, clip, efficientnet_pytorch, lmdb; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
`cuda True`가 나와야 GPU 학습 가능. `False`거나 torch 설치 에러면 → CUDA 빌드로 교체:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # 예: CUDA 12.1 (드라이버에 맞게 cu124 등)
```

**자주 쓰는 conda 명령**
```bash
conda deactivate            # 환경 빠져나오기
conda env list              # 환경 목록 (frodo_lan 있는지)
conda activate frodo_lan    # 다시 들어가기
conda env remove -n frodo_lan   # 환경 삭제(재생성하고 싶을 때)
```

> 원저자의 전체 환경(`environment_mbra.yml`)은 무겁고 원본 전체 학습(diffusion/NoMaD)용입니다. 언어 파인튜닝만
> 할 거면 `environment_frodo_lan.yml`로 충분합니다 — `diffusion_policy`/`map_cache`/`warmup_scheduler`/`wandb`는
> 코드에서 가드되어 없어도 `lan_only_ft`가 돌아갑니다(`use_wandb: False`가 기본).

### 3-2. 체크포인트
`omnivla-edge.pth`(HF: `NHirose/omnivla-edge`)를 `train/omnivla-edge.pth`에 둡니다. config의
`load_edge_ckpt: ./omnivla-edge.pth`가 `train/` 기준 상대경로이므로, 다른 곳에 두면 그 경로로 바꾸세요.

### 3-3. ⭐ 데이터 경로 수정 (여기만 고치면 됩니다)
`train/config/frodo_lan_ft.yaml`의 `datasets_lan.frodo_lan`에서 `image`/`pickle`은 **데이터 루트**
(=`episode_*` 폴더들의 상위)로 **둘 다 같은 값**, `train`/`test`는 **캐시용 빈 폴더**로 설정합니다.
현재 이 레포의 `omnivla_dataset/` 기준으로 이미 아래처럼 채워져 있습니다:

```yaml
datasets_lan:
  frodo_lan:
    image:  /home/shy/suhyeon/OmniVLA_edge/omnivla_dataset/               # 데이터 루트(episode_* 상위)
    pickle: /home/shy/suhyeon/OmniVLA_edge/omnivla_dataset/               # 같은 루트
    train:  /home/shy/suhyeon/OmniVLA_edge/omnivla_dataset/_cache/train/  # LMDB 캐시(빈 폴더)
    test:   /home/shy/suhyeon/OmniVLA_edge/omnivla_dataset/_cache/test/
```

다른 컴퓨터/위치에서 돌리려면 이 4개 경로만 바꾸면 됩니다 — `image`·`pickle`은 항상 **같은 값(데이터 루트)**,
`train`·`test`는 서로 다른 빈 폴더. 캐시 폴더는 `mkdir -p <루트>/_cache/{train,test}`로 만들어 두세요.

> 이 4줄 + (필요 시) `load_edge_ckpt` 외에는 건드릴 필요 없습니다.

### 3-4. 실행
```bash
cd train
python train.py -c config/frodo_lan_ft.yaml
```

**정상 동작 신호**
- 시작 로그: `[ckpt] loaded. missing=0 unexpected=0` → `omnivla-edge.pth` 정상 로드.
- `[params] trainable X.XM / total Y.YM` → 백본 freeze로 학습 파라미터가 줄어든 것 확인.
- 배치 로그: `total=… action=… obj=…` → `action` loss가 감소하면 학습 중.
- epoch마다: `[eval] ... test_action_loss=.. base_divergence(image-goal)=..`
- 저장: `train/logs_frodo_lan_ft/best.pth`(최적) + `latest.pth`.
- 결과 체크포인트는 원본과 동일 구조라 공개 레포 `inference/run_omnivla_edge.py`에 그대로 넣어 추론 가능.

### 3-5. 🔥 스모크 테스트 (본 학습 전에 먼저 하세요)
이 코드는 **end-to-end로 아직 검증되지 않았습니다.** 데이터를 조금(예: **에피소드 2~3개**)만 준비해
먼저 **파이프라인이 도는지** 확인하는 걸 강력 권장합니다. 목적은 "성능"이 아니라 **버그 잡기**입니다.

스모크 테스트용으로 config를 잠깐 이렇게:
```yaml
batch_size: 8       # 데이터가 적으면 24 -> 8 (drop_last=True라 배치가 0개 되는 것 방지)
epochs: 2           # 2~3이면 충분
freeze_backbone: True
```
그리고 **확인할 것만** 봅니다:
- `[ckpt] loaded. missing=0 unexpected=0` 나오나
- 첫 배치에서 크래시 없이 `total=/action=/obj=` loss가 찍히나
- `[eval]`·체크포인트 저장까지 도나

⚠️ 소량 데이터라 `test_action_loss`·`base_divergence` **수치는 노이즈**입니다 — "돌아가는지"만 보고,
**"좋아졌는지"는 판단하지 마세요.** 파이프라인이 정상이면, 데이터가 충분히 모인 뒤 원래 하이퍼파라미터로
본 학습을 돌리면 됩니다.

> 참고 — 에피소드 경계: 현재 `frodo_lan` 로더는 데이터를 하나의 연속 시퀀스로 보고 context를 인덱스
> 인접(`iv-1..iv-5`)으로 가져옵니다. 여러 에피소드를 `0..N`으로 이어붙이면 **경계에서 context가 이전
> 에피소드를 물어올 수 있습니다.** 스모크 테스트엔 무해하지만, 본 학습에선 에피소드별 분리/경계 처리가
> 필요할 수 있습니다(§6 참고).

### 3-6. 프리셋 (소량 언어전용 vs 대용량 범용)
두 개의 config를 준비해뒀습니다. **데이터만 `omnivla_dataset/`에 더 넣고 config만 고르면** 됩니다.

**두 프리셋 모두 범용(언어 + 이미지-goal)** 을 학습합니다. 차이는 **데이터 규모에 맞춘 하이퍼파라미터**(백본 freeze / lr / epochs / split)뿐입니다.

| | `config/frodo_lan_ft.yaml` (소량, 보수적) | `config/frodo_lan_ft_full.yaml` (대용량) |
|---|---|---|
| **학습 모달리티** | 언어/pose + 이미지-goal (mask 6·7·8) | 언어/pose + 이미지-goal (mask 6·7·8) |
| `use_image_goal` / teacher | **True** / MBRA (`mbra.pth`) | **True** / MBRA (`mbra.pth`) |
| `freeze_backbone` | **True** (시각 인코더 고정) | **False** (전체 학습) |
| `lr` | 2e-5 | 1e-4 |
| `epochs` | 12 | 40 |
| `early_stop_patience` | 3 | 5 |
| `split_by_episode` | **False** (인덱스 90/10) | **True** (에피소드 통째 held-out) |
| `output_dir` | `./logs_frodo_lan_ft` | `./logs_frodo_lan_ft_full` |
| prompt 필터 / cap | ON / **cap 없음** (MBRA 켜짐) | ON / **cap 없음** (MBRA 켜짐) |

- **`split_by_episode`**: 소량일 땐 False가 유리 — 에피소드 단위로 빼면 (3개 중 1개=33%처럼) 학습 데이터가 크게 줄기 때문. 대용량일 땐 True로 두면 test 에피소드가 train과 완전히 분리돼 **지표 신뢰도**가 높아짐(train/test 누수 없음). 캐시는 전략·프레임수별로 따로 생성되어 전환 시 자동 재빌드됩니다.

- **소량(에피소드 몇 개)** → `frodo_lan_ft.yaml` (백본 고정·저LR로 과적합 방지).
- **데이터 충분** → `frodo_lan_ft_full.yaml` (백본까지 당신 도메인에 맞춰 학습).
- 두 프리셋 다 **MBRA 체크포인트 필요** (아래 3-7). 데이터 경로(`datasets_lan`)는 두 파일에서 동일하게 유지하세요.
- **cap**: `use_image_goal: True`이면 이미지-goal이 학습 대상이므로 `base_divergence` cap은 **자동 비활성**입니다. 순수 언어전용으로 돌리고 싶을 때만 `use_image_goal: False` + `base_divergence_max: 0.5`로 base 주행 보호 cap을 켤 수 있습니다.

### 3-7. 범용(이미지-goal) 학습 — MBRA teacher 준비
LeLaN 데이터는 원본 OmniVLA에서 **두 가지 역할**로 쓰입니다: (1) 언어/pose → 궤적 라벨(`nomad_traj_norm`, 우리 데이터에 이미 있음), (2) **이미지-goal → MBRA teacher가 런타임에 생성**하는 궤적. `frodo_lan_ft_full.yaml`은 이 둘을 모두 학습하는 **범용 모델**입니다.

- **이미지-goal(mask 6)**: loader가 20% 확률로 같은 에피소드의 **미래 프레임**을 goal 이미지로 뽑고(`goal_id>0`), 그 궤적 타겟을 MBRA(`ExAug_dist_delay`)가 생성합니다. 언어/pose(mask 7·8, `goal_id==0`)는 그대로 `nomad_traj_norm`을 씁니다.
- **타겟 혼합**: `action_ref = (goal_id==0 ? nomad_traj_norm : MBRA_traj)` — 원본 OmniVLA/edge와 동일.
- **base_divergence cap 없음**: 이미지-goal이 이제 **학습 대상**이므로(부작용 이탈이 아니라), 언어전용 경로의 divergence cap을 이 경로에는 적용하지 않습니다.

**MBRA 체크포인트 다운로드** (약 395MB, `ExAug_dist_delay` 가중치):
```bash
cd train
wget "https://huggingface.co/NHirose/MBRA/resolve/main/mbra.pth" -O mbra.pth
```
- config의 `load_mbra: ./mbra.pth`가 이 파일을 가리킵니다. 다른 곳에 두면 경로만 바꾸세요.
- 우리 레포의 `ExAug_dist_delay`에 **정확히 로드**됩니다(검증됨: missing=0, unexpected=0). MBRA "데이터"나 코드베이스 clone은 **필요 없습니다** — 가중치 파일 하나면 됩니다.
- `*.pth`는 `.gitignore`에 있어 커밋되지 않습니다.

### 체크포인트 — 실행마다 시각 폴더로 저장 (덮어쓰기 없음) ✅
- `timestamp_run: True`(기본)이면 각 실행이 **`output_dir/<시작시각>/`** 하위 폴더에 저장됩니다
  (예: `logs_frodo_lan_ft/2026_07_15_14_03_21/best.pth`). 그래서 **재실행해도 이전 결과를 덮어쓰지 않습니다.**
  이전 파일을 지울 필요가 전혀 없습니다.
- 재학습은 **항상 원본 `omnivla-edge.pth`(base)에서 시작**합니다(이전 파인튜닝을 이어받지 않음) — 남은 파일이 새 학습을 오염시키지 않습니다.
- 한 실행 폴더 안에서 `best.pth`(최저 test loss) + `latest.pth`가 저장됩니다. 시각 폴더는 이름순 정렬되므로 가장 최근 실행이 맨 아래.
- 매번 같은 폴더에 저장하고 싶으면(옛 동작) config에서 `timestamp_run: False`.
- 원본 `omnivla-edge.pth`는 어느 경우에도 **읽기 전용**, 절대 안 바뀝니다.

---

## 4. 파인튜닝 전략 — 보수적 부분 파인튜닝 (기본값)

베이스 모델은 **대규모 멀티로봇 데이터**로 학습된 강한 prior를 갖는데 커스텀 데이터는 극소량이므로,
전 파라미터를 흔드는 풀 파인튜닝은 **과적합 + catastrophic forgetting** 위험이 큽니다. 그래서 기본값은
**부분 파인튜닝**입니다.

**학습 루프** `train_lan_only_ft` / `main_lan_only_ft`:
1. LeLaN 배치에서 obs(context+현재), 객체 crop, CLIP 이미지, 객체 pose, prompt, `nomad_traj_norm`을 받음.
2. CLIP로 prompt → 텍스트 특징. 위성지도/맵 채널은 0(언어 조건이므로).
3. goal mask `{7=언어만, 8=언어+GPS}` 샘플 → 모델 forward.
4. 손실: `action_loss`(궤적 vs `nomad_traj_norm`, 주 손실) + `obj_loss`(마지막 waypoint vs 객체 pose) + dist/smooth.
5. **teacher 호출 없음.** CLIP은 freeze.

**증강**: 공개 OmniVLA(`prismatic/vla/datasets/lelan_dataset.py`)와 동일하게 좌우 반전(flip) 증강을
적용합니다 — 이미지 좌우 미러 + 궤적 `nomad_traj_norm`의 y(col1)·sin(col3) 부호 반전 + 객체 pose의
좌(left) 부호 반전 (forward·cos는 유지). 소량 데이터의 일반화에 도움.
> NoMaD 궤적을 **학습 중 pickle에서 그대로 불러오는(런타임 NoMaD 실행 없음)** 방식도 공개 OmniVLA와 동일함을 코드 대조로 확인했습니다.

**`freeze_backbone: True`**(기본): 시각 인코더 `obs_encoder`/`goal_encoder`/`goal_encoder_img`를 freeze
(BN 통계까지 eval로 고정). 학습되는 건 `decoder`·`film_model`·`compress_goal_enc_lan`·`local_goal`·
`action_predictor`·`dist_predictor`뿐. 시작 시 `[params] trainable X.XM / total Y.YM` 로 확인.

**낮은 LR + early stopping**: `lr=2e-5`, `epochs=12`, `warmup_epochs=2`, `early_stop_patience=3`.
매 epoch 자동 분할된 **test(10%)에서 `test_action_loss`를 평가**하고, 개선이 없으면 조기 종료.
`best.pth`(최저 test loss) + `latest.pth` 저장.

### 기본 주행이 망가졌는지 확인하는 회귀 검사 ⭐
다른 내비 데이터가 없어도, **우리 이미지에 이미지-goal 모달리티(mask 6)를 먹여 base 모델과 파인튜닝 모델의
출력이 얼마나 벌어졌는지**를 매 epoch 측정합니다:
```
[eval] epoch k  test_action_loss=..  test_obj_loss=..  base_divergence(image-goal)=X.XXXX
```
- `base_divergence(image-goal)` ≈ 0 → 이미지-goal 기본 주행이 **보존됨**.
- 이 값이 epoch가 갈수록 **크게 증가** → 언어 학습이 공유 디코더를 통해 기본 주행을 **망가뜨리는 중**.
  이 경우 LR을 더 낮추거나, epoch를 줄이거나, `decoder`도 일부 freeze하세요.
- 원본 `omnivla-edge.pth`(base)는 학습 내내 frozen 사본으로 메모리에 유지되어 비교 기준이 됩니다.

순수 언어만 학습하려면 `train_utils.py`의 `random.choice([7, 8])`를 `[7]`로 바꾸세요.
풀 파인튜닝을 원하면 config에 `freeze_backbone: False`.

---

## 5. 성능 비교 (base vs 파인튜닝) — `eval_compare.py`

학습과 별개로, **원본과 파인튜닝본을 test 분할에서 나란히 비교**하는 독립 스크립트입니다.
두 체크포인트 파일 모두 수정하지 않습니다.

```bash
cd train
python eval_compare.py                                   # base=config, ft=logs_frodo_lan_ft/best.pth
python eval_compare.py --ft ./logs_frodo_lan_ft/latest.pth
python eval_compare.py --base ./omnivla-edge.pth --ft ./logs_frodo_lan_ft/best.pth
```

출력 예:
```
=== Language-goal (mask 7) metrics on frodo_lan TEST split ===
metric                  base    finetuned        delta   better
action_mse            0.xxxx      0.xxxx      -0.xxxx   (lower)  <-- improved
waypoint_err_m        0.xxxx      0.xxxx      -0.xxxx   (lower)  <-- improved
endpoint_err_m        0.xxxx      0.xxxx      -0.xxxx   (lower)
object_err_m          0.xxxx      0.xxxx      -0.xxxx   (lower)
heading_cos           0.xxxx      0.xxxx      +0.xxxx   (higher) <-- improved

=== Basic-driving regression: image-goal (mask 6) divergence, base vs fine-tuned ===
  base_divergence(image-goal) = 0.xxxx
```
- **언어 지시 수행이 좋아졌나** → `action_mse`/`waypoint_err_m`/`endpoint_err_m`/`object_err_m`가 base보다 **낮으면** 개선, `heading_cos`는 **높으면** 개선.
- **기본 주행이 망가졌나** → `base_divergence(image-goal)`가 **작으면** 이미지-goal 주행이 보존된 것.
- `waypoint_err_m`/`endpoint_err_m`/`object_err_m`는 **미터 단위**(정규화 0.12 반영)라 직관적으로 해석됩니다.

> 지표는 `nomad_traj_norm`(데이터의 궤적 라벨) 기준입니다. 궁극적으로는 실제 로봇/시뮬레이터에서의
> 정성 평가가 가장 신뢰도 높습니다.

### 궤적 시각화 — `visualize_traj.py`
수치 대신 **실제 예측 궤적을 눈으로** 비교합니다. 왼쪽에 장면 이미지 + **객체 bbox(빨간 박스)** + prompt,
오른쪽에 조감도 궤적(GT · base · 파인튜닝 · 객체 위치 ★ · 로봇 ■). 데이터셋 코드에 의존하지 않는 독립 스크립트입니다.

```bash
cd train
# 기본: base vs logs_frodo_lan_ft/best.pth
python visualize_traj.py

# 특정/여러 체크포인트 비교, 샘플 수·출력 지정
python visualize_traj.py --ft logs_frodo_lan_ft/2026_07_15_10_51_28/best.pth --n 6 --out compare.png
python visualize_traj.py --ft runA/best.pth --ft runB/best.pth        # 여러 개(반복)

# 특정 에피소드에서만 뽑기
python visualize_traj.py --episode episode_0037 --n 6
```
옵션: `--base`(기본 `omnivla-edge.pth`), `--ft`(반복 가능), `--episode`, `--n`, `--seed`, `--out`.
결과 PNG로 **파인튜닝이 GT 궤적을 잘 따라가는지 / 궤적이 객체 방향을 향하는지**를 바로 확인할 수 있습니다.
(생성 PNG `eval_*.png`는 `.gitignore` 처리됨.)

### 임의 이미지로 추론 — `infer.py`
**라벨/데이터셋 형식 없이** 아무 이미지 + 언어 prompt만으로 궤적을 예측합니다 (다른 데이터로 inference 확인용).
언어 모달리티(goal mask 7)에선 모델이 **관측 이미지 + 텍스트만** 쓰므로 GPS/지도/이미지-goal은 dummy(0)로 넣습니다.

```bash
cd train
# 현재 이미지 1장 + 지시
python infer.py --prompt "go to the metal gate" --images cur.jpg

# context 시퀀스 (오래된 -> 최신, 마지막이 현재 프레임)
python infer.py --prompt "turn toward the white wall" --images t-2.jpg t-1.jpg t.jpg

# 다른 체크포인트 / 출력 파일
python infer.py --prompt "..." --images cur.jpg --ckpt logs_frodo_lan_ft/best.pth --out out.png
```
→ 콘솔에 8-스텝 waypoint(전방·좌 m) 출력 + `infer_out.png`(장면 | 예측 궤적) 저장. 한 번의 forward(결정론적).
> 추론 자체엔 `nomad_traj` 라벨이 필요 없습니다(라벨은 평가/비교용). 그래서 어떤 데이터의 이미지든 바로 넣어볼 수 있습니다.
> 단, 모델은 학습 데이터의 prompt 분포(객체 묘사)에 맞춰져 있어, 전혀 다른 표현/도메인엔 반응이 약할 수 있습니다.

## 6. 알려진 문제 / 주의점

### 이 레포 전반 (원본 정리 전 코드)
- **그대로는 원본 전체 학습(`omnivla_edge.yaml`) 불가**:
  - `train.py`가 `../diffusion_policy`를, `train_utils.py`가 `map_cache`를 import → 둘 다 레포에 없음.
    (→ 우리가 **import 가드**를 넣어 `lan_only_ft` 경로는 이들 없이도 import됨. 하지만 diffusion/NoMaD를 쓰는 원본 경로는 여전히 이 모듈들이 필요.)
  - config의 데이터/체크포인트 경로가 전부 원저자 서버(`/nfs/kun2/...`, `/home/noriaki/...`) 기준.
  - `train.py`에 wandb API 키가 하드코딩(`use_wandb: False`로 비활성화 권장).
  - `path_save_load`, `path_mapcache`가 하드코딩된 절대경로.

### 우리 `lan_only_ft` 경로의 미검증 지점
- **End-to-end 미실행**: 구문/구조만 검증. 첫 실행 시 collate/이미지 로드 등에서 사소한 수정 가능성.
- **좌표 프레임**: `nomad_traj_norm`과 pose를 `(x=전방, y=좌)`로 가정. 첫 실행 때 `image_log` 또는 몇 배치의
  `action_pred` vs `nomad_traj_norm`을 시각적으로 대조해 확인 권장.
- **obj_loss 스케일**: 보조항(`obj_loss`, 가중치 0.05)에서 `metric_waypoint_spacing=0.125`(코드) vs 데이터 `0.12`로 ~4% 차이.
  주 손실(`action_loss`)엔 영향 없음.
- 데이터가 작으면(예: 수백 프레임) 과적합 주의 — 먼저 §3-5 스모크 테스트로 검증할 것.
- **에피소드 경계 / 분할** (해결됨): 로더가 `episode_*` 폴더를 인식해 context(`iv-1..iv-5`)를 **같은
  에피소드로 클램프**하고, `split_by_episode: True`면 test를 **에피소드 단위로 통째 held-out**합니다
  (train/test 누수 방지). 소량 데이터에선 `False`(인덱스 90/10)가 학습 데이터를 더 확보해 유리합니다.

---

## 7. 원본 전체 학습을 하려면 (참고)

`lan_only_ft` 없이 원본 멀티모달 학습(`config/omnivla_edge.yaml`)을 돌리려면 추가로 필요:
1. `diffusion_policy`, `map_cache` 모듈 확보
2. frodobot / GNM / LeLaN / BDD 데이터셋 + 경로 수정
3. teacher 체크포인트: `exaug_labeler.pth`(ExAug/MBRA), `nomad_crop.pth`(NoMaD), `exaug_bdd_labeler.pth`
4. 위성지도 타일 캐시

이건 이 레포의 목표(내 데이터로 언어 파인튜닝) 범위 밖입니다.

---

## 라이선스 / 출처
원본: [NHirose/OmniVLA_edge](https://github.com/NHirose/OmniVLA_edge), [NHirose/OmniVLA](https://github.com/NHirose/OmniVLA).
`LICENSE` 파일 참조.
