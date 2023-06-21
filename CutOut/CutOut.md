::: {.cell .markdown}
# Cutout data augmentation

In this notebook, we will reproduce the results of the paper

> DeVries, T. and Taylor, G.W., 2017. Improved regularization of convolutional neural networks with Cutout. arXiv preprint [arXiv:1708.04552](https://arxiv.org/abs/1708.04552). 

We will use the author's implementation of their technique, from [https://github.com/uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout), which is licensed under an Educational Community License version 2.0.

:::



::: {.cell .markdown}

## Learning outcomes

After working through this notebook, you should be able to:

* Describe how Cutout works as a regularization technique,
* Enumerate specific claims (both quantitative claims, qualitative claims, and claims about the underlying mechanism behind a result) from the Cutout paper,
* Execute experiments (following the given procedure) to try and validate each claim about Cutout data augmentation,
* Evaluate whether your own result matches quantitative claims in the Cutout paper (i.e. whether it is within the confidence intervals for each reported numeric result),
* Evaluate whether your own result validates qualitative claims in the Cutout paper,
* Evaluate whether your own results support the author's claim about the underlying mechanism behind the result.

:::


::: {.cell .code}
``` python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid
from torchvision import datasets, transforms


import math
import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt
import pdb
import argparse
from tqdm import tqdm
import os
```
:::


::: {.cell .code}
``` python
# note: This notebook has been developed and tested for pytorch 
print(torch. __version__)
```
:::



::: {.cell .markdown }

## Cutout as a regularization technique

This Jupyter notebook is designed to illustrate the implementation and
usage of the Cutout data augmentation technique in deep learning,
specifically in the context of Convolutional Neural Networks (CNNs).

:::

::: {.cell .markdown }


Cutout is a regularization and data augmentation technique for
convolutional neural networks (CNNs). It involves randomly masking out
square regions of input during training. This helps to improve the
robustness and overall performance of CNNs by encouraging the network to
better utilize the full context of the image, rather than relying on the
presence of a small set of specific visual features.

Cutout is computationally efficient as it can be applied during data
loading in parallel with the main training task. It can be used in
conjunction with existing forms of data augmentation and other
regularizers to further improve model performance.

The technique has been evaluated with state-of-the-art architectures on
popular image recognition datasets such as CIFAR-10, CIFAR-100, and
SVHN, often achieving state-of-the-art or near state-of-the-art results.
:::


::: {.cell .markdown}
In the following cells, we will see how Cutout works when applied to a sample image.

<!-- To do: explain the code with reference to section 3.2. Implementation Details -->
In the code provided above, we see a Python class named Cutout defined. This class is designed to apply the Cutout data augmentation technique to an image. Below is an explanation of the class and its methods:

- The Cutout class is initialized with two parameters:

    - `n_holes`: the number of patches to cut out of each image.
    - `length`: the length (in pixels) of each square patch.
- The `__call__` method implements the Cutout technique. This method takes as input a tensor `img` representing an image, and returns the same image with `n_holes` number of patches of dimension `length` x `length` cut out of it.

Here's a step-by-step explanation of what's happening inside the `__call__` method:

1. The method first retrieves the height h and width w of the input image.

2. A mask is then initialized as a 2D numpy array of ones with the same dimensions as the input image.

3. The method then enters a loop which runs for n_holes iterations. In each iteration:

    - A pair of coordinates y and x are randomly selected within the height and width of the image.

    - The method then calculates the coordinates of a square patch around the (y, x) coordinate. The patch has a length of length pixels, and the method ensures that the patch doesn't fall outside the image by using the np.clip function.

    - The corresponding area in the mask is set to zero.

4. The mask is then converted to a PyTorch tensor and expanded to the same number of channels as the input image.

5. Finally, the method applies the mask to the input image, effectively setting the pixels in the masked regions to zero, and returns the result.

Remember to import necessary libraries like numpy (np) and PyTorch (torch) before running this class definition. The class Cutout can then be used as part of your data augmentation pipeline when training your models.

The Cutout code we are using comes from this specific file in the original GitHub repository: [https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py].
:::



::: {.cell .code}
``` python
# to do: link to the file in the original repo that this comes from
# Source Code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
```
:::


::: {.cell .markdown}

To see how it works, in the following cell, you will upload an image of your choice to this workspace:

<!-- to do - add instructions for uploading image on Colab, or on Chameleon -->
To see how Cutout works, let's upload an image and apply Cutout to it. Follow these steps to upload an image in this Google Colab notebook:

1. Click on the folder icon in the left sidebar to open the 'Files' tab.
2. Click the 'Upload to session storage' button (the icon looks like a file with an up arrow).
3. Select the image file from your local machine that you want to upload.
4. Wait for the upload to finish. The uploaded file should now appear in the 'Files' tab.
After the image is uploaded, we can use Python code to load it into our notebook and apply the Cutout augmentation

If you are using Chameleon, here are the steps:
<!-- to do - add instructions for uploading image on Chameleon -->
1. Click on the upload icon in the left sidebar.
2. Select the image file from your local machine that you want to upload.
3. Wait for the upload to finish. The uploaded file should now appear in the 'Files' tab.
After the image is uploaded, we can use Python code to load it into our notebook and apply the Cutout augmentation to the image.
:::



::: {.cell .code}
```python
# TODO: Replace 'sample.png' with the filename of your own image. 
# If your image is inside a directory, include the directory's name in the path.
img = Image.open('/content/sample.png')
```
:::


::: {.cell .markdown}
Then, the following cell will display your image directly, without any data augmentation:
:::


::: {.cell .code }
``` python
# Convert the image to a PyTorch tensor
img_tensor = transforms.ToTensor()(img)

# Display the original image
plt.figure(figsize=(6,6))
plt.imshow(img_tensor.permute(1, 2, 0))
plt.show()
```
:::

::: {.cell .markdown}
and the next cell will display your image with Cutout applied:
:::


::: {.cell .code}
``` python
# Create a Cutout object
Cutout = Cutout(n_holes=1, length=300)

# Apply Cutout to the image
img_tensor_Cutout = Cutout(img_tensor)

# Convert the tensor back to an image for visualization
img_Cutout = transforms.ToPILImage()(img_tensor_Cutout)

# Display the image with Cutout applied
plt.figure(figsize=(6,6))
plt.imshow(img_tensor_Cutout.permute(1, 2, 0))
plt.show()
```
:::


::: {.cell .markdown}

Things to try:

* You can re-run the cell above several times to see how the occlusion is randomly placed in a different position each time.
* You can try changing the `length` parameter in the cell above, and re-running, to see how the size of the occlusion can change.
* You can try changing the `n_holes` parameter in the cell above, and re-running, to see how the number of occlusions can change.

:::


::: {.cell .code}
``` python
 #TODO: Set the number of patches ("holes") to cut out of the image.
n_holes = 

#TODO: Set the size (length of a side) of each patch.
length = 


# Create a Cutout object
Cutout = Cutout(n_holes, length)

# Apply Cutout to the image
img_tensor_Cutout = Cutout(img_tensor)

# Convert the tensor back to an image for visualization
img_Cutout = transforms.ToPILImage()(img_tensor_Cutout)

# Display the image with Cutout applied
plt.figure(figsize=(6,6))
plt.imshow(img_tensor_Cutout.permute(1, 2, 0))
plt.show()
```
:::


::: {.cell .markdown}

Cutout was introduced as an alternative to two closely related techniques:

* Data Augmentation for Images:  Data augmentation is a strategy used to increase the diversity of the data available for training models, without actually collecting new data. For image data, this could include operations like rotation, scaling, cropping, flipping, and adding noise. The goal is to make the model more robust by allowing it to see more variations of the data.

* Dropout in Convolutional Neural Networks: Dropout is a regularization technique for reducing overfitting in neural networks. During training, some number of layer outputs are randomly ignored or "dropped out". This has the effect of making the layer look-like and be treated-like a layer with a different number of nodes and connectivity to the prior layer. In effect, dropout simulates ensembling a large number of neural networks with different architectures, which makes the model more robust.

<!-- to do - expand on these -->

:::

::: {.cell .markdown}
In the following code snippet, we demonstrate some "standard" data augmentation techniques commonly used in image preprocessing. These techniques include random horizontal flipping, random cropping, and color jittering (random variation in brightness, contrast, saturation, and hue). The augmented image is then displayed alongside the original image for comparison.

:::
::: {.cell .code}
```python
# to do - show the same image with "standard" data augmentation techniques
# discussed in the related work section of the paper

# Import necessary libraries
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ColorJitter

# Define standard data augmentation techniques
transforms = transforms.Compose([
    RandomHorizontalFlip(),
    RandomCrop(size=(28, 28), padding=4),  # assuming input image is size 28x28
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

# Apply transformations to the image
augmented_img = transforms(img)

# Display the original and augmented image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(augmented_img)
ax[1].set_title('Augmented Image')
plt.show()

```
:::




::: {.cell .markdown}
## Identifying claims from the Cutout paper

To reproduce the results from the original Cutout paper, we will first need to identify the specific, falsifiable claims in that paper, by reading it very carefully. Then, we will design experiments to validate each claim. 

These claims may be quantitative (i.e. describe a specific numeric result), qualitative (i.e. describe a general characteristic of the result), or they may relate to the mechanism behind a result (i.e. describe *why* a particular result occurs).

<!-- to do - go through the paper, quote little snippets and explain each claim and organize them -->

### Qualitative Claims
- 'We aimed to remove maximally activated
features in order to encourage the network to consider less
prominent features.' 

### Quantitative Claims
#### ResNet18

Test error (%, flip/translation augmentation, mean/std normalization, mean of 5 runs) 

| **Network** | **CIFAR-10** | **CIFAR-100** |
| ----------- | ------------ | ------------- |
| ResNet18    | 4.72         | 22.46         |
| ResNet18 + cutout | 3.99   | 21.96         |  


#### WideResNet

WideResNet model implementation from https://github.com/xternalz/WideResNet-pytorch  

Test error (%, flip/translation augmentation, mean/std normalization, mean of 5 runs)  

| **Network** | **CIFAR-10** | **CIFAR-100** | **SVHN** |
| ----------- | ------------ | ------------- | -------- |
| WideResNet  | 3.87         | 18.8          | 1.60     |
| WideResNet + cutout | 3.08 | 18.41         | **1.30** |


#### Shake-shake Regularization Network

Shake-shake regularization model implementation from https://github.com/xgastaldi/shake-shake

Test error (%, flip/translation augmentation, mean/std normalization, mean of 3 runs)  

| **Network** | **CIFAR-10** | **CIFAR-100** |
| ----------- | ------------ | ------------- |
| Shake-shake | 2.86         | 15.58         |
| Shake-shake + cutout | 2.56 | 15.20 |



:::


::: {.cell .markdown}
## Execute experiments to validate quantitative and qualitative claims

:::



::: {.cell .markdown}
### Implement Cutout on CIFAR10 Dataset
:::


::: {.cell .code}
``` python
if not os.path.exists('/content/checkpoints'):
    os.makedirs('/content/checkpoints')
```
:::



::: {.cell .code}
``` python
# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #,
    #Cutout(n_holes=1, length=16)  # Cutout applied here
])

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
```
:::

::: {.cell .code}
``` python
# Load the CIFAR-10 dataset with transformations applied
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
```
:::

::: {.cell .code}
``` python
# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images before Cutout
imshow(torchvision.utils.make_grid(images))
```
:::

::: {.cell .code}
``` python
# Apply Cutout and show images after
Cutout_images = torch.stack([Cutout(n_holes=1, length=16)(img) for img in images])
imshow(torchvision.utils.make_grid(Cutout_images))
```
:::

::: {.cell .markdown}
### 4. Methods and Implementation {#4-methods-and-implementation}
:::

::: {.cell .markdown}
### ResNet Code
:::

::: {.cell .code}
``` python
# ResNet

# From https://github.com/uoguelph-mlrg/Cutout/blob/master/model/resnet.py

'''ResNet18/34/50/101/152 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(self.expansion*planes))

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = conv3x3(3,64)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512*block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
      return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


def ResNet18(num_classes=10):
  return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes=10):
  return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes=10):
  return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes=10):
  return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes=10):
  return ResNet(Bottleneck, [3,8,36,3], num_classes)

def test_resnet():
  net = ResNet50()
  y = net(Variable(torch.randn(1,3,32,32)))
  print(y.size())

# test_resnet()
```
:::

::: {.cell .markdown}
### WideResNet Code
:::

::: {.cell .code}
``` python
# WideResNet

# From https://github.com/uoguelph-mlrg/Cutout/blob/master/model/wide_resnet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out
```
:::

::: {.cell .code}
``` python
class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
```
:::

::: {.cell .markdown}
### 5. Model Training and Evaluation {#5-model-training-and-evaluation}
:::

::: {.cell .code}
``` python
# run train.py --dataset cifar10 --model resnet18 --data_augmentation --Cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --Cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --Cutout --length 20

'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
'''

#from util.misc import CSVLogger
#from util.Cutout import Cutout

#from model.resnet import ResNet18
#from model.wide_resnet import WideResNet
'''
model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--Cutout', action='store_true', default=False,
                    help='apply Cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

args = parser.parse_args()
'''

def main(args):
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  cudnn.benchmark = True  # Should make training should go faster for large models


  torch.manual_seed(args.seed)
  if args.cuda:
      torch.cuda.manual_seed(args.seed)

  test_id = args.dataset + '_' + args.model

  print(args)

  # Image Preprocessing
  if args.dataset == 'svhn':
      normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                      std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
  else:
      normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                      std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

  train_transform = transforms.Compose([])
  if args.data_augmentation:
      train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
      train_transform.transforms.append(transforms.RandomHorizontalFlip())
  train_transform.transforms.append(transforms.ToTensor())
  train_transform.transforms.append(normalize)
  if args.Cutout:
      train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


  test_transform = transforms.Compose([
      transforms.ToTensor(),
      normalize])

  if args.dataset == 'cifar10':
      num_classes = 10
      train_dataset = datasets.CIFAR10(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

      test_dataset = datasets.CIFAR10(root='data/',
                                      train=False,
                                      transform=test_transform,
                                      download=True)
  elif args.dataset == 'cifar100':
      num_classes = 100
      train_dataset = datasets.CIFAR100(root='data/',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

      test_dataset = datasets.CIFAR100(root='data/',
                                      train=False,
                                      transform=test_transform,
                                      download=True)
  elif args.dataset == 'svhn':
      num_classes = 10
      train_dataset = datasets.SVHN(root='data/',
                                    split='train',
                                    transform=train_transform,
                                    download=True)

      extra_dataset = datasets.SVHN(root='data/',
                                    split='extra',
                                    transform=train_transform,
                                    download=True)

      # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
      data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
      labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
      train_dataset.data = data
      train_dataset.labels = labels

      test_dataset = datasets.SVHN(root='data/',
                                  split='test',
                                  transform=test_transform,
                                  download=True)

  # Data Loader (Input Pipeline)
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=2)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2)

  if args.model == 'resnet18':
      cnn = ResNet18(num_classes=num_classes)
  elif args.model == 'wideresnet':
      if args.dataset == 'svhn':
          cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                          dropRate=0.4)
      else:
          cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                          dropRate=0.3)

  cnn = cnn.cuda()
  criterion = nn.CrossEntropyLoss().cuda()
  cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                  momentum=0.9, nesterov=True, weight_decay=5e-4)

  if args.dataset == 'svhn':
      scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
  else:
      scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

  filename = 'logs/' + test_id + '.csv'
  csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


  def test(loader):
      cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
      correct = 0.
      total = 0.
      for images, labels in loader:
          images = images.cuda()
          labels = labels.cuda()

          with torch.no_grad():
              pred = cnn(images)

          pred = torch.max(pred.data, 1)[1]
          total += labels.size(0)
          correct += (pred == labels).sum().item()

      val_acc = correct / total
      cnn.train()
      return val_acc


  for epoch in range(args.epochs):

      xentropy_loss_avg = 0.
      correct = 0.
      total = 0.

      progress_bar = tqdm(train_loader)
      for i, (images, labels) in enumerate(progress_bar):
          progress_bar.set_description('Epoch ' + str(epoch))

          images = images.cuda()
          labels = labels.cuda()

          cnn.zero_grad()
          pred = cnn(images)

          xentropy_loss = criterion(pred, labels)
          xentropy_loss.backward()
          cnn_optimizer.step()

          xentropy_loss_avg += xentropy_loss.item()

          # Calculate running average of accuracy
          pred = torch.max(pred.data, 1)[1]
          total += labels.size(0)
          correct += (pred == labels.data).sum().item()
          accuracy = correct / total

          progress_bar.set_postfix(
              xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
              acc='%.3f' % accuracy)

      test_acc = test(test_loader)
      tqdm.write('test_acc: %.3f' % (test_acc))

      scheduler.step(epoch)  # Use this line for PyTorch <1.4
      # scheduler.step()     # Use this line for PyTorch >=1.4

      row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
      csv_logger.writerow(row)

  torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
  csv_logger.close()
```
:::

::: {.cell .code}
``` python
class Args:
    dataset = 'cifar10'
    model = 'resnet18'
    batch_size = 128
    epochs = 200
    learning_rate = 0.1
    data_augmentation = False
    Cutout = False
    n_holes = 1
    length = 16
    no_cuda = False
    seed = 0

args = Args()
```
:::

::: {.cell .code}
``` python
main(args)
```
:::

::: {.cell .markdown}
## Evaluate your results for qualitative and quantitative claims
:::



::: {.cell .markdown}
## Execute experiments to validate the suggested mechanism

:::

::: {.cell .markdown}
## Evaluate your results for validating the suggested mechanism
### Implementing GradCam
:::

