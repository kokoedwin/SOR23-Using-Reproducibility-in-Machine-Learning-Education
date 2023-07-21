
::: {.cell .markdown}
### ResNet
:::


::: {.cell .markdown}
Note: for faster training, use Runtime > Change Runtime Type to run this notebook on a GPU.
:::


::: {.cell .markdown}

In the Cutout paper, the authors claim that:

1. Cutout improves the robustness and overall performance of convolutional neural networks.
2. Cutout can be used in conjunction with existing forms of data augmentation and other regularizers to further improve model performance.

In this section, we will evaluate these claims using a ResNet model. For the ResNet model, the specific quantitative claims are given in the following table:

Test error (%, flip/translation augmentation, mean/std normalization, mean of 5 runs) and “+” indicates standard data augmentation (mirror
+ crop)

| **Network** | **CIFAR-10** | **CIFAR-10+** | **CIFAR-100** | **CIFAR-100+** |
| ----------- | ------------ | ------------- | ------------ | ------------- |
| ResNet18    | 10.63         | 4.72        | 36.68         | 22.46         |
| ResNet18 + cutout | 9.31   | 3.99         | 34.98         | 21.96        |  


The provided table displays the results of experiments conducted on the CIFAR-10 and CIFAR-100 datasets using the ResNet18 architecture, revealing the impact of standard and cutout data augmentation techniques. The "CIFAR-10+" and "CIFAR-100+" labels indicate the use of standard data augmentation, which involves mirror and crop techniques.

With the use of standard data augmentation on CIFAR-10, the ResNet18 model's test error is significantly reduced from 14.04% to 5.72%. Further enhancement is achieved when cutout augmentation is applied, bringing down the error to 4.86%. A similar pattern is observed in the case of the CIFAR-100 dataset, where standard augmentation reduces the ResNet18 model's test error from 40.13% to 24.36%. Upon applying cutout augmentation, a slight further reduction in error to 23.9% is noted.

These findings emphasize the efficacy of both standard and cutout data augmentation techniques in enhancing the model's performance, evidenced by the reduction in test error rates on both CIFAR-10 and CIFAR-100 datasets. The results also highlight that the impact of data augmentation can vary based on the complexity of the dataset, illustrated by the differing rates of error reduction between CIFAR-10 and CIFAR-100.

:::



::: {.cell .markdown}

#### Implementation

:::


::: {.cell .code}
```python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
```
:::


::: {.cell .markdown}
This code block is used for creating a directory named 'checkpoints'. This directory will be used to store the weights of our models, which are crucial for both preserving our progress during model training and for future use of the trained models.

Creating such a directory and regularly saving model weights is a good practice in machine learning, as it ensures that you can resume your work from where you left off, should the training process be interrupted.
:::

::: {.cell .code}
``` python
# Create file names 'checkpoints' to save the weight of the models
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
```
:::

::: {.cell .markdown}
### 4.1.1 ResNet Code
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
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

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

def test_resnet():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test_resnet()
```
:::



::: {.cell .markdown}
### 4.1.3 Model Evaluate Test Code
This function evaluates the performance of the model on a given data loader (loader). It sets the model to evaluation mode (eval), calculates the accuracy on the dataset, and returns the validation accuracy. It then switches the model back to training mode (train) before returning the validation accuracy.
:::

::: {.cell .code}
``` python
def test(loader, cnn):
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
```
:::

::: {.cell .markdown}
Check Cuda GPU availability and set seed number
:::
::: {.cell .code}
``` python
cuda = torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

seed = 1
torch.manual_seed(seed)
```
:::

::: {.cell .markdown}
Image Processing for CIFAR-10
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])

train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize_image)



test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_image])
```
:::