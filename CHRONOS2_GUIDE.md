# Chronos-2 사용 가이드 (Chronos-2 Usage Guide)

이 문서는 Chronos-2 모델을 fine-tuning하고 representation을 추출하는 방법을 설명합니다.

This document explains how to fine-tune Chronos-2 and extract representations from it.

---

## 📚 목차 (Table of Contents)

1. [Chronos-2 Fine-tuning 방법](#chronos-2-fine-tuning)
2. [Representation 추출 방법](#representation-extraction)
3. [코드 예제](#code-examples)

---

## 🎯 Chronos-2 Fine-tuning

### 주요 파일 및 모듈 (Key Files & Modules)

| 파일 (File) | 용도 (Purpose) |
|------------|---------------|
| `src/chronos/chronos2/pipeline.py` | **메인 API** - `Chronos2Pipeline.fit()` 메서드 포함 |
| `src/chronos/chronos2/trainer.py` | 배치 처리를 위한 커스텀 트레이너 |
| `src/chronos/chronos2/dataset.py` | 데이터 전처리를 위한 데이터셋 클래스 |
| `src/chronos/chronos2/config.py` | 모델 설정 |

### Fine-tuning 모드 (Fine-tuning Modes)

Chronos-2는 두 가지 fine-tuning 모드를 지원합니다:

1. **Full Fine-tuning** - 모든 모델 파라미터 업데이트
2. **LoRA Fine-tuning** - Low-Rank Adaptation (파라미터 효율적)

### 기본 사용 예제 (Basic Usage)

```python
from chronos import Chronos2Pipeline
import torch

# 사전 학습된 모델 로드 (Load pretrained model)
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# 시계열 데이터 준비 (Prepare time series data)
# 리스트 형태의 1D 텐서들
inputs = [
    torch.rand(100),  # 길이 100인 시계열
    torch.rand(120),  # 길이 120인 시계열
    torch.rand(95)    # 길이 95인 시계열
]

# Full fine-tuning
finetuned_pipeline = pipeline.fit(
    inputs=inputs,
    prediction_length=24,          # 예측 길이
    num_steps=1000,                # 학습 스텝 수
    learning_rate=1e-6,            # 학습률
    batch_size=256,                # 배치 크기 (시계열 개수)
    finetune_mode="full",          # "full" 또는 "lora"
)

# Fine-tuned 모델로 예측
forecasts = finetuned_pipeline.predict(inputs, prediction_length=24)
```

### LoRA Fine-tuning 예제 (Advanced)

```python
from peft import LoraConfig

# 커스텀 LoRA 설정
lora_config = LoraConfig(
    r=8,                    # LoRA rank
    lora_alpha=16,          # LoRA alpha
    target_modules=[        # LoRA를 적용할 모듈들
        "self_attention.q",
        "self_attention.v",
        "self_attention.k",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ],
)

finetuned_pipeline = pipeline.fit(
    inputs=inputs,
    prediction_length=24,
    num_steps=1000,
    learning_rate=1e-5,            # LoRA에는 더 높은 학습률 권장
    finetune_mode="lora",
    lora_config=lora_config,
    batch_size=256,
)
```

### 주요 파라미터 (Key Parameters)

| 파라미터 (Parameter) | 설명 (Description) | 기본값 (Default) |
|---------------------|-------------------|-----------------|
| `prediction_length` | 예측 길이 (forecast horizon) | **필수** (required) |
| `num_steps` | 학습 스텝 수 | 1000 |
| `learning_rate` | 학습률 (optimizer learning rate) | 1e-6 (full), 1e-5 권장 (LoRA) |
| `batch_size` | 배치 크기 | 256 |
| `context_length` | 최대 과거 길이 | 모델 기본값 (~2048) |
| `validation_inputs` | 검증 데이터 (optional) | None |
| `min_past` | 최소 과거 타임스텝 수 | prediction_length |
| `output_dir` | 체크포인트 저장 디렉토리 | None |
| `finetune_mode` | Fine-tuning 모드 | "full" |

### 지원하는 입력 형식 (Supported Input Formats)

```python
# 1. 간단한 텐서 (Simple tensors)
inputs = torch.rand(4, 256)  # (batch_size, history_length)

# 2. 텐서 리스트 (List of tensors)
inputs = [torch.rand(100), torch.rand(150), torch.rand(200)]

# 3. 공변량을 포함한 딕셔너리 리스트 (List of dicts with covariates)
inputs = [
    {
        "target": torch.rand(100),
        "past_covariates": {"temperature": torch.rand(100)},
        "future_covariates": {"temperature": torch.rand(24)},
    },
    {
        "target": torch.rand(150),
        "past_covariates": {"temperature": torch.rand(150)},
    },
]
```

---

## 🔍 Representation Extraction

### Representation 추출 방법

Chronos-2에서 representation(임베딩)을 추출하는 두 가지 방법:

#### 1. Pipeline의 `embed()` 메서드 사용 (권장)

```python
from chronos import Chronos2Pipeline
import torch

# 모델 로드
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# 시계열 데이터
time_series_data = [
    torch.rand(100),
    torch.rand(150),
    torch.rand(200),
]

# Representation 추출
embeddings, loc_scales = pipeline.embed(
    inputs=time_series_data,
    batch_size=256,
    context_length=None  # 모델 기본값 사용
)

# embeddings의 shape: (n_series, n_variates, num_patches + 2, d_model)
print(f"Embeddings shape: {embeddings.shape}")
print(f"d_model (representation size): {embeddings.shape[-1]}")
```

#### 2. Model의 `encode()` 메서드 사용 (저수준 API)

```python
# 모델 직접 접근
model = pipeline.model

# 데이터 준비 (더 복잡한 전처리 필요)
encoder_outputs, loc_scale, _, _ = model.encode(
    context=context_tensor,
    context_mask=context_mask,
    group_ids=group_ids,
    num_output_patches=num_patches
)

# Hidden states (representations) 추출
hidden_states = encoder_outputs[0]  # Chronos2EncoderOutput의 last_hidden_state
```

### Representation 크기 및 차원 (Representation Dimensions)

| 차원 (Dimension) | 값 (Value) | 설명 (Description) |
|-----------------|-----------|-------------------|
| **d_model** | **512** | 임베딩 차원 (기본값) |
| **시계열당 Shape** | `(n_variates, num_patches + 2, d_model)` | 각 시계열의 representation shape |
| **num_patches** | `context_length / input_patch_size` | 입력 패치 개수 |
| **추가 토큰** | +2 | [REG] 토큰 + 마스크된 출력 패치 토큰 |

### 기본 모델 설정 (Default Model Configuration)

```python
# src/chronos/chronos2/config.py에서 확인 가능
d_model = 512              # 임베딩 차원
d_kv = 64                  # 키-값 프로젝션 크기
num_layers = 6             # 인코더 레이어 수
num_heads = 8              # 어텐션 헤드 수
input_patch_size = 16      # 입력 패치 크기
output_patch_size = 16     # 출력 패치 크기
context_length = 2048      # 최대 컨텍스트 길이
```

### 아키텍처 세부사항 (Architecture Details)

Chronos-2 인코더는 다음과 같이 처리합니다:

1. **입력 패치** → `input_patch_embedding` ResidualBlock을 통해 임베딩
2. **6개의 인코더 블록** 통과, 각 블록은:
   - Time Self-Attention (시간적 의존성)
   - Group Self-Attention (그룹 내 교차 시계열 의존성)
   - Feed Forward 레이어
3. **최종 출력** → `last_hidden_state` shape: `(batch_size * n_variates, num_patches + 2, 512)`

### Representation 활용 예제 (Using Representations)

```python
# Representation 추출
embeddings, loc_scales = pipeline.embed(inputs=time_series_data)

# 평균 풀링으로 시계열 레벨 임베딩 생성
# Shape: (n_series, n_variates, num_patches + 2, 512)
# -> (n_series, 512)
series_embeddings = embeddings.mean(dim=(1, 2))

# 유사도 계산
from torch.nn.functional import cosine_similarity
similarity = cosine_similarity(series_embeddings[0], series_embeddings[1], dim=0)
print(f"Series similarity: {similarity.item()}")

# 클러스터링, 분류 등 다운스트림 태스크에 활용 가능
```

---

## 📖 Code Examples

### 완전한 Fine-tuning 예제 (Complete Fine-tuning Example)

```python
import pandas as pd
import torch
from chronos import Chronos2Pipeline

# 1. 모델 로드
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    torch_dtype=torch.float16  # 메모리 효율성을 위한 half precision
)

# 2. 데이터 준비
# DataFrame 형식
context_df = pd.DataFrame({
    'id': ['A'] * 100 + ['B'] * 100,
    'timestamp': pd.date_range('2020-01-01', periods=100).tolist() * 2,
    'target': torch.rand(200).tolist(),
    'temperature': torch.rand(200).tolist(),  # 공변량
})

# 3. Fine-tuning
finetuned_pipeline = pipeline.fit(
    inputs=context_df,
    prediction_length=24,
    num_steps=1000,
    learning_rate=1e-6,
    batch_size=256,
    finetune_mode="full",
    output_dir="./checkpoints",
    validation_inputs=None,  # 검증 데이터가 있다면 여기에
)

# 4. 예측
predictions = finetuned_pipeline.predict(
    context=context_df,
    prediction_length=24,
    num_samples=100,  # 확률적 예측을 위한 샘플 수
)

print(f"Predictions shape: {predictions.shape}")
# Shape: (n_series, num_samples, prediction_length)
```

### Representation 기반 전이 학습 예제 (Transfer Learning with Representations)

```python
import torch
import torch.nn as nn
from chronos import Chronos2Pipeline

# 1. Chronos-2로 representation 추출
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# 시계열 데이터
train_data = [torch.rand(100) for _ in range(1000)]
labels = torch.randint(0, 5, (1000,))  # 5-class classification

# Embeddings 추출
embeddings, _ = pipeline.embed(inputs=train_data)
# Shape: (1000, 1, num_patches + 2, 512)

# 평균 풀링
features = embeddings.mean(dim=(1, 2))  # (1000, 512)

# 2. 분류기 학습
classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 5)
).cuda()

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 간단한 학습 루프
for epoch in range(10):
    optimizer.zero_grad()
    outputs = classifier(features.cuda())
    loss = criterion(outputs, labels.cuda())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 3. 추론
test_data = [torch.rand(100) for _ in range(100)]
test_embeddings, _ = pipeline.embed(inputs=test_data)
test_features = test_embeddings.mean(dim=(1, 2))

with torch.no_grad():
    predictions = classifier(test_features.cuda())
    predicted_classes = predictions.argmax(dim=1)

print(f"Predicted classes: {predicted_classes}")
```

---

## 🔗 참고 자료 (References)

- **Chronos-2 논문**: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- **Chronos-2 모델**: [HuggingFace](https://huggingface.co/amazon/chronos-2)
- **Chronos-2 Quickstart 노트북**: `notebooks/chronos-2-quickstart.ipynb`
- **소스 코드**:
  - Pipeline: `src/chronos/chronos2/pipeline.py`
  - Model: `src/chronos/chronos2/model.py`
  - Trainer: `src/chronos/chronos2/trainer.py`
  - Config: `src/chronos/chronos2/config.py`

---

## ❓ FAQ

### Q1: Full fine-tuning과 LoRA 중 어느 것을 선택해야 하나요?

**A**: 
- **Full fine-tuning**: 데이터가 충분하고 계산 리소스가 있을 때, 최고 성능
- **LoRA**: 데이터가 적거나 메모리/계산 리소스가 제한적일 때, 파라미터 효율적

### Q2: Representation의 차원(512)을 변경할 수 있나요?

**A**: 모델의 `d_model` 파라미터는 사전 학습 시 결정되며, 사전 학습된 모델을 사용할 때는 변경할 수 없습니다. 처음부터 학습한다면 `Chronos2CoreConfig`에서 변경 가능합니다.

### Q3: 다변량(multivariate) 시계열도 지원하나요?

**A**: 네, Chronos-2는 단변량, 다변량, 공변량을 포함한 예측을 모두 지원합니다. 입력 데이터를 딕셔너리 형식으로 제공하면 됩니다.

### Q4: Fine-tuning 후 모델을 저장하려면?

**A**: `output_dir` 파라미터를 지정하면 체크포인트가 자동으로 저장됩니다. 또는:
```python
finetuned_pipeline.model.save_pretrained("./my_finetuned_model")
```

---

## 📝 요약 (Summary)

### Chronos-2 Fine-tuning
- **메서드**: `Chronos2Pipeline.fit()`
- **모드**: Full 또는 LoRA
- **주요 파라미터**: `prediction_length`, `num_steps`, `learning_rate`, `finetune_mode`

### Representation 추출
- **메서드**: `Chronos2Pipeline.embed()`
- **크기**: **512 차원** (d_model)
- **Shape**: `(n_series, n_variates, num_patches + 2, 512)`
- **활용**: 유사도 분석, 클러스터링, 분류 등 다운스트림 태스크

이 가이드가 Chronos-2 모델 활용에 도움이 되기를 바랍니다! 🚀
