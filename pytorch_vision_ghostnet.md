---
layout: hub_detail
background-class: hub-background
body-class: hub
title: GhostNet
summary: Efficient networks by generating more features from cheap operations
category: researchers
image: ghostnet.png
author: Huawei Noah's Ark Lab
tags: [vision, scriptable]
github-link: https://github.com/huawei-noah/ghostnet
github-id: huawei-noah/ghostnet
featured_image_1: ghostnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/GhostNet
---

```python
import torch
model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
model.eval()
```

모든 사전 학습된 모델들은 입력 이미지가 동일한 방식으로 정규화 되는 것을 요구합니다. 
다시 말해 `H`와 `W`가 적어도 `224`이고, `(3 x H x W)`의 shape를 가지는 3채널 RGB 이미지들의 미니배치를 말합니다.
이 이미지들은 `[0, 1]`의 범위로 로드되어야 하고, `mean = [0.485, 0.456, 0.406]`
과 `std = [0.229, 0.224, 0.225]`를 사용하여 정규화되어야 합니다.

Here's a sample execution.

```python
# pytorch 웹사이트에서 예시 이미지를 다운로드
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# Download ImageNet labels
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

고스트넷 아키텍처는 다양한 특징 맵을 효율적인 연산으로 생성하는 고스트 모듈 구조로 이루어집니다. 고유한 특징 맵을 기반으로 한 효율적인 연산으로 추론에 기반이 되는 고유 특징들을 충분하게 드러내는 고스트 특징 맵을 생성합니다. 벤치마크에서 수행된 실험을 통해 속도와 정확도의 상충 관계에 관한 고스트넷의 우수성을 보여줍니다.

사전 학습된 모델을 사용한 ImageNet 데이터셋에 따른 정확도는 아래에 나열되어 있습니다.

| Model structure | FLOPs       | Top-1 acc   | Top-5 acc   |
| --------------- | ----------- | ----------- | ----------- |
|  GhostNet 1.0x  | 142M        | 73.98       | 91.46       |


### 참고

다음 [링크](https://arxiv.org/abs/1911.11907)에서 논문의 전체적인 내용에 대하여 읽을 수 있습니다.


>@inproceedings{han2019ghostnet,
>    title={GhostNet: More Features from Cheap Operations},
>    author={Kai Han and Yunhe Wang and Qi Tian and Jianyuan Guo and Chunjing Xu and Chang Xu},
>    booktitle={CVPR},
>    year={2020},
>}

