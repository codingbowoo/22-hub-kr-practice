---
layout: hub_detail
background-class: hub-background
body-class: hub
title: RoBERTa
summary: A Robustly Optimized BERT Pretraining Approach
category: researchers
image: fairseq_logo.png
author: Facebook AI (fairseq Team)
tags: [nlp]
github-link: https://github.com/pytorch/fairseq/
github-id: pytorch/fairseq
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/RoBERTa
---


### 모델 설명

Bidirectional Encoder Representations from Transformers, [BERT][1]는 텍스트에서 의도적으로 숨겨진(masked) 부분을 예측하는 학습에 획기적인 self-supervised pretraining 기술이다. 결정적으로 BERT가 학습한 표현은 downstream tasks에 잘 일반화되는 것으로 나타났으며, BERT가 처음 출시된 2018년에 많은 NLP benchmark datasets에서 state-of-the-art 결과를 달성했다.

[RoBERTa][2]는 BERT의 language masking strategy를 기반으로 구축되며, BERT의 next-sentence pretraining objective를 제거하고 훨씬 더 큰 미니 배치와 학습 속도로 훈련하는 등 주요 하이퍼파라미터를 수정한다. 또한 RoBERTa는 더 오랜 시간 동안 BERT보다 훨씬 많은 데이터에 대해 학습되었다. 이를 통해 RoBERTa의 표현은 BERT와 비교해 downstream tasks을 훨씬 잘 일반화할 수 있다.


### 요구 사항

전처리 과정을 위해 추가적인 Python 의존성이 필요합니다.

```bash
pip install regex requests hydra-core omegaconf
```


### 예시

##### Load RoBERTa
```python
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Apply Byte-Pair Encoding (BPE) to input text
```python
tokens = roberta.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
assert roberta.decode(tokens) == 'Hello world!'
```

##### Extract features from RoBERTa
```python
# Extract the last layer's features
last_layer_features = roberta.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
```

##### Use RoBERTa for sentence-pair classification tasks
```python
# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

with torch.no_grad():
    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 0  # contradiction

    # Encode another pair of sentences
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 2  # entailment
```

##### Register a new (randomly initialized) classification head
```python
roberta.register_classification_head('new_task', num_classes=3)
logprobs = roberta.predict('new_task', tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
```


### 참고

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding][1]
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach][2]


[1]: https://arxiv.org/abs/1810.04805
[2]: https://arxiv.org/abs/1907.11692
