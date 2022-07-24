
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
github-id: pytorch/vision
featured_image_1: resnext.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/ResNext
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
# or
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
model.eval()
```

모든 사전훈련된 모델은 입력 이미지가 같은 방식으로 정규화되었다고 가정합니다.
즉, 미니배치(mini-batch)의 3채널 RGB 이미지들은 `(3 x H x W)`의 shape을 가지며, `H`와 `W`는 최소 `224`이상이어야 하며, 각 이미지들은 `[0, 1]`의 범위에서 로드되어야 하며, 그 다음 `mean = [0.485, 0.456, 0.406]` 과 `std = [0.229, 0.224, 0.225]`를 이용해 정규화되어야 합니다. 
아래 예시 코드가 있습니다.

```python
# 파이토치 웹 사이트에서 다운로드한 이미지 입니다.
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 예시 코드 (torchvision 필요)
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 가정하는대로 미니배치 생성

# gpu를 사용할 수 있다면, 속도를 위해 입력과 모델을 gpu로 옮김
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# output은 shape가 [1000]인 Tensor 자료형이며, 이는 Imagenet 데이터셋의 각 클래스에 대한 모델의 확신도(confidence)를 나타냄.
print(output[0])
# output은 정규화되지 않았으므로, 확률화하기 위해 softmax 함수를 처리합니다.
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

Resnext models were proposed in [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).
Here we have the 2 versions of resnet models, which contains 50, 101 layers repspectively.
A comparison in model archetechure between resnet50 and resnext50 can be found in Table 1.
Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

|  Model structure  | Top-1 error | Top-5 error |
| ----------------- | ----------- | ----------- |
|  resnext50_32x4d  | 22.38       | 6.30        |
|  resnext101_32x8d | 20.69       | 5.47        |

### References

 - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
