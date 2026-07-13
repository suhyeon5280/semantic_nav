# semantic_nav — OmniVLA-edge 언어 파인튜닝

이 레포는 [NHirose/OmniVLA_edge](https://github.com/NHirose/OmniVLA_edge)(정리 전 연구 코드)를 포크해서,
**커스텀 LeLaN 형식 데이터로 OmniVLA-edge 모델을 언어(instruction) 조건으로 파인튜닝**할 수 있도록
전용 경로(`lan_only_ft`)를 추가한 것입니다.

> ⚠️ 이 문서는 코드 분석에 기반한 정리입니다. 아래 "언어 전용 파인튜닝" 경로는 구조·구문은 검증했지만
> **데이터가 있는 GPU 머신에서 end-to-end로는 아직 실행/검증되지 않았습니다.** 첫 실행 시 사소한 수정이
>필요할 수 있습니다.

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

```
<DATA_ROOT>/frodo_lan/
├── image/00000000.jpg, 00000001.jpg, …   # 224×224 RGB, 8자리 zero-pad, 0부터 연속(시간순)
├── pickle/00000000.pkl, 00000001.pkl, …  # image와 1:1, 같은 인덱스
├── train/     # 빈 폴더(쓰기 가능) — 학습 시 LMDB 캐시가 여기 생성됨
└── test/      # 빈 폴더(쓰기 가능)
```

**규칙**
- 이미지는 **반드시 224×224**. 로더가 리사이즈하지 않고 bbox/crop이 224 기준이기 때문(코드에서 안전하게 224로 강제하긴 함).
- 프레임은 **시간순**(과거 프레임을 context로, 미래 프레임을 goal로 참조).
- 객체 없는 프레임도 `pickle.dump([], f)`로 pkl 저장(파일 크기 0이면 스킵됨).
- train/test 폴더는 **비워 둠**. 분할은 로더가 인덱스 비율 **90/10**으로 자동 수행.

**pickle 형식** (프레임당 객체 리스트, 각 원소 dict):
```python
{
  "bbox":             np.array([[top, bottom, left, right]]),  # (1,4) int, 224 공간
  "pose_median":      np.array([[forward, left]]),             # (1,2) float, meters
  "pose_median_norm": np.array([[forward, left]]) / 0.12,      # (1,2)
  "nomad_traj_norm":  np.array (8,4) float32,                  # 누적 (x=fwd, y=left, cos, sin), 0.12 정규화
  "prompt":           ["asphalt road", "paved surface", ...],  # 문자열 리스트
}
```

---

## 3. 실행 방법

```bash
# 1) 환경 (레포 루트)
conda env create -f environment_mbra.yml
conda activate <env>
# 누락 시: pip install efficientnet_pytorch openai-clip lmdb diffusers warmup_scheduler prettytable utm

# 2) 체크포인트: omnivla-edge.pth 를 train/ 에 배치 (HF: NHirose/omnivla-edge)

# 3) config 경로 수정: train/config/frodo_lan_ft.yaml 의 datasets_lan.frodo_lan 4개 경로
#    (train/test/image/pickle) 를 실제 <DATA_ROOT> 로 (끝에 '/' 필수)

# 4) 실행
cd train
python train.py -c config/frodo_lan_ft.yaml
```

**정상 동작 신호**
- 시작 로그: `[ckpt] loaded. missing=0 unexpected=0` → `omnivla-edge.pth` 정상 로드.
- 배치 로그: `total=… action=… obj=…` → `action` loss가 감소하면 학습 중.
- 저장: `train/logs_frodo_lan_ft/latest.pth`(에폭마다) + `0.pth, 1.pth, …`.
- 결과 `latest.pth`는 원본과 동일 구조라 공개 레포 `inference/run_omnivla_edge.py`에 그대로 넣어 추론 가능.

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
- 데이터가 작으면(예: 수백 프레임) 과적합 주의 — 스모크 테스트/소규모 파인튜닝용.

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
