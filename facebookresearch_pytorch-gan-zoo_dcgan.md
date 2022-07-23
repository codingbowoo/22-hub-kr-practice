---
layout: hub_detail
background-class: hub-background
body-class: hub
title: DCGAN on FashionGen
summary: A simple generative image model for 64x64 images
category: researchers
image: dcgan_fashionGen.jpg
author: FAIR HDGAN
tags: [vision, generative]
github-link: https://github.com/facebookresearch/pytorch_GAN_zoo/blob/master/models/DCGAN.py
github-id: facebookresearch/pytorch_GAN_zoo
featured_image_1: dcgan_fashionGen.jpg
featured_image_2: no-image
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/DCGAN_on_fashiongen
order: 10
---

```python
import torch
use_gpu = True if torch.cuda.is_available() else False

model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)
```

모델에 입력하는 노이즈 벡터의 크기는 `(N, 120)` 이며 여기서 `N`은 생성하고자 하는 이미지의 개수입니다.
데이터 생성은 `.buildNoiseData` 함수를 사용하여 구성될 수 있습니다.
모델의 `.test` 함수를 사용하면 노이즈 벡터를 입력받아 이미지를 생성합니다.

```python
num_images = 64
noise, _ = model.buildNoiseData(num_images)
with torch.no_grad():
    generated_images = model.test(noise)

# let's plot these images using torchvision and matplotlib
import matplotlib.pyplot as plt
import torchvision
plt.imshow(torchvision.utils.make_grid(generated_images).permute(1, 2, 0).cpu().numpy())
# plt.show()
```

왼쪽에 있는것과 유사한 이미지를 살펴볼 수 있습니다.

만약 본 예제의 GANs과 다른 본인만의 DCGAN을 학습시키고 싶다면, 이곳을 살펴보세요 [PyTorch GAN Zoo](https://github.com/facebookresearch/pytorch_GAN_zoo).

### Model Description

In computer vision, generative models are networks trained to create images from a given input. In our case, we consider a specific kind of generative networks: GANs (Generative Adversarial Networks) which learn to map a random vector with a realistic image generation.

DCGAN is a model designed in 2015 by Radford et. al. in the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). It is a GAN architecture both very simple and efficient for low resolution image generation (up to 64x64).



### Requirements

- Currently only supports Python 3

### References

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
