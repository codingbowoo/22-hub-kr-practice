---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ResNet
summary: Deep residual networks pre-trained on ImageNet
category: researchers
image: resnet.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
github-id: pytorch/vision
featured_image_1: resnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/ResNet
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#  Layer의 수에 따라 달라지는 Resnet 모델들.
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()
```

모든 사전학습 된(pre-trained) 모델들은 인풋 이미지 같은 방법으로 정규화된다. 3개의 채널을 가진 RGB 미니배치는 H와 W가 최소 224 이상인 (3 x H x W)의 모양을 가진다.  
이미지는 [0, 1]범위로 로드한 다음  mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225]을 통해 정규화된다. 

다음은 실행한 예제이다. 

```python
# 파이토치 웹사이트에서 예제 이미지를 다운로드 받기 
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 실행 예제(torchvision을 필요로함)
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

# 가능하다면 속도를 위해 입력과 모델을 GPU로 옮기기
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# 결과는 정규화되지 않은 점수를 가집니다. 확률을 얻기 위해서는 소프트맥스로 돌리세요.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# 이미지넷 라벨들을 다운로드 
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리들 읽기 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 각각의 이미지별로 탑 카테고리를 보이기 
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 묘사

“이미지 인식을 위한 잔차 딥러닝 ([Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf))”에서 Resnet 모델들이 제안되었다. 각각 18, 34, 50, 101, 152개의 층을 가진 5개 버전의 resnet 모델이 있다. 상세 모델 구조는 1번 표에서 찾을 수 있다. 모델들의 1-crop 에러 비율은 선행 학습된(pre-trained) 이미지넷 데이터셋 밑에 나열돼있다.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  resnet18       | 30.24       | 10.92       |
|  resnet34       | 26.70       | 8.58        |
|  resnet50       | 23.85       | 7.13        |
|  resnet101      | 22.63       | 6.44        |
|  resnet152      | 21.69       | 5.94        |

### 참고자료

 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
