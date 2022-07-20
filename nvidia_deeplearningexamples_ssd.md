---
layout: hub_detail
background-class: hub-background
body-class: hub
title: SSD
summary: Single Shot MultiBox Detector model for object detection
category: researchers
image: nvidia_logo.png
author: NVIDIA
tags: [vision]
github-link: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
github-id: NVIDIA/DeepLearningExamples
featured_image_1: ssd_diagram.png
featured_image_2: ssd.png
accelerator: cuda
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/SSD
---


### Model Description

SSD300 모델은 "단일 심층 신경망을 사용하여 이미지에서 물체를 감지하는 방법"으로 설명 하는 [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) 논문을 기반으로 합니다. 입력 크기는 300x300으로 고정되어 있습니다.

이 모델과 논문에 설명된 모델의 큰 차이점은 백본에 있습니다. 논문에서 사용한 VGG 모델은 더 이상 사용되지 않으며 ResNet-50 모델로 대체되었습니다.

[Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012) 논문에서 , 백본에 대해 다음과 같은 개선이 이루어졌습니다.:

*   conv5_x, avgpool, fc 및 softmax 레이어는 기존의 분류 모델에서 제거되었습니다.
*   conv4_x의 모든 strides는 1x1로 설정됩니다.

백본 뒤에는 5개의 컨볼루션 레이어가 추가됩니다. 또한 컨볼루션 레이어 외에도 6개의 detection heads를 추가했습니다.
The backbone is followed by 5 additional convolutional layers.
In addition to the convolutional layers, we attached 6 detection heads:
*   첫 번째 detection head는 마지막 conv4_x 레이어에 연결됩니다.
*   나머지 5개의 detection head는 추가되는 5개의 컨볼루션 레이어에 부착됩니다.

Detector heads는 논문에서 언급된 것과 유사하지만, 각각의 컨볼루션 레이어 뒤에 BatchNorm 레이어를 추가함으로써 성능이 향상됩니다.

### Example

In the example below we will use the pretrained SSD model to detect objects in sample images and visualize the result.

To run the example you need some extra python packages installed. These are needed for preprocessing images and visualization.
```bash
pip install numpy scipy scikit-image matplotlib
```

Load an SSD model pretrained on COCO dataset, as well as a set of utility methods for convenient and comprehensive formatting of input and output of the model.
```python
import torch
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
```

Now, prepare the loaded model for inference
```python
ssd_model.to('cuda')
ssd_model.eval()
```

Prepare input images for object detection.
(Example links below correspond to first few test images from the COCO dataset, but you can also specify paths to your local images here)
```python
uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg',
    'http://images.cocodataset.org/val2017/000000037777.jpg',
    'http://images.cocodataset.org/val2017/000000252219.jpg'
]
```

Format the images to comply with the network input and convert them to tensor.
```python
inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs)
```

Run the SSD network to perform object detection.
```python
with torch.no_grad():
    detections_batch = ssd_model(tensor)
```

By default, raw output from SSD network per input image contains
8732 boxes with localization and class probability distribution.
Let's filter this output to only get reasonable detections (confidence>40%) in a more comprehensive format.
```python
results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
```

The model was trained on COCO dataset, which we need to access in order to translate class IDs into object names.
For the first time, downloading annotations may take a while.
```python
classes_to_labels = utils.get_coco_object_dictionary()
```

Finally, let's visualize our detections
```python
from matplotlib import pyplot as plt
import matplotlib.patches as patches

for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
plt.show()
```

### Details
For detailed information on model input and output,
training recipies, inference and performance visit:
[github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)
and/or [NGC](https://ngc.nvidia.com/catalog/resources/nvidia:ssd_for_pytorch)

### References

 - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper
 - [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012) paper
 - [SSD on NGC](https://ngc.nvidia.com/catalog/resources/nvidia:ssd_for_pytorch)
 - [SSD on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)
