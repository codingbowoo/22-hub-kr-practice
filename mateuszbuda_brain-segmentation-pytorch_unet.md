---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: U-Net for brain MRI
summary: U-Net with batch normalization for biomedical image segmentation with pretrained weights for abnormality segmentation in brain MRI
image: unet_tcga_cs_4944.png
author: mateuszbuda
tags: [vision]
github-link: https://github.com/mateuszbuda/brain-segmentation-pytorch
github-id: mateuszbuda/brain-segmentation-pytorch
featured_image_1: unet_brain_mri.png
accelerator: "cuda-optional"
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/U-NET-for-brain-MRI
---

```python
import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

```

위 코드는 뇌 MRI 볼륨 데이터 셋 [kaggle.com/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)의 이상 탐지를 위해 사전 학습된 U-Net 모델을 불러옵니다. 
사전 학습된 모델은 첫 번째 계층에서 3개의 입력 채널, 1개의 출력 채널 그리고 32개의 특징을 가져야 합니다.

### 모델 설명

This U-Net model comprises four levels of blocks containing two convolutional layers with batch normalization and ReLU activation function, and one max pooling layer in the encoding part and up-convolutional layers instead in the decoding part.
The number of convolutional filters in each block is 32, 64, 128, and 256.
The bottleneck layer has 512 convolutional filters.
From the encoding layers, skip connections are used to the corresponding layers in the decoding part.
Input image is a 3-channel brain MRI slice from pre-contrast, FLAIR, and post-contrast sequences, respectively.
Output is a one-channel probability map of abnormality regions with the same size as the input image.
It can be transformed to a binary segmentation mask by thresholding as shown in the example below.

### 예시

사전 학습된 모델에 입력되는 이미지는 3개의 채널을 가져야 하며 256x256 픽셀로 크기 조정 및 각 볼륨마다 z-점수로 정규화된 상태여야 합니다.

```python
# 예시 이미지 다운로드 
import urllib
url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
import numpy as np
from PIL import Image
from torchvision import transforms

input_image = Image.open(filename)
m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

print(torch.round(output[0]))
```

### 참고문헌 

- [Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm](http://arxiv.org/abs/1906.03720)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Brain MRI segmentation dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
