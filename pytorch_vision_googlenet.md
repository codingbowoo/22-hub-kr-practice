---
layout: hub_detail
background-class: hub-background
body-class: hub
title: GoogLeNet
summary: GoogLeNet was based on a deep convolutional neural network architecture codenamed "Inception" which won ImageNet 2014.
category: researchers
image: googlenet1.png
author: Pytorch Team
tags: [vision]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
github-id: pytorch/vision
featured_image_1: googlenet1.png
featured_image_2: googlenet2.png
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/GoogleNet
order: 10
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()
```

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.
다시 학습받은 모든 모델은 이미지를 같은 방법으로 인풋 이미지를 정규화할려고 기대한다.
즉 작은 집단의 3종류의 RGB이미지의 모양은 `(3 x H x W)'이고, `H` 와 `W` 적어도 '224' 가 되길 기대한다.
그 이미지들은 `[0, 1]` 범위안에 실려야하고, 그리고  `mean = [0.485, 0.456, 0.406]`와 `std = [0.229, 0.224, 0.225]`을 보통으로 사용한다.
@@


Here's a sample execution.
예시 예제가 있다


```python
# Download an example image from the pytorch website
# 파이토치 웹사이트에서 예시 그림을 다운받아라 
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# sample execution (requires torchvision)
# 실행 예제 ('torchvision'을 요구한다)
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
                                        # 모델에서 요구되는 미니-배치를 생성하라
# move the input and model to GPU for speed if available
# 속도를 위해 가능하다면 'input'과 'model'을 'GPU'로 옮겨라
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# 'Imagenet'의 1000 classes 보다 높은 confidence scores와 함께 'Tensor'의 모양을 1000으로 하여라
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# 결과는 비정상적인 scores다. 개연성을 가질려면 softmax를 실행 할 수도 있다
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


### Model Description

GoogLeNet was based on a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The 1-crop error rates on the ImageNet dataset with a pretrained model are list below.

### 모델 설명

'GoogLeNet'은 "Inception"이라고 암호화된 이름으로 나선형의 심층 신경망의 아키텍쳐에 기반은 둔 새로운상태의 예술의 분류와 'ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014)'의 발견에 책임이있다

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  googlenet       | 30.22       | 10.47       |



### References

 - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
### 참조

 - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) 