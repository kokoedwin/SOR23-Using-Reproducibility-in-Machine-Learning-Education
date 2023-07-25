
::: {.cell .markdown}
# 03. WideResNet
:::

::: {.cell .markdown}

WideResNet model implementation from https://github.com/xternalz/WideResNet-pytorch  

Note: for faster training, use Runtime > Change Runtime Type to run this notebook on a GPU.
:::


:
::: {.cell .markdown}

In the Cutout paper, the authors claim that:

1. Cutout improves the robustness and overall performance of convolutional neural networks.
2. Cutout can be used in conjunction with existing forms of data augmentation and other regularizers to further improve model performance.

In this section, we will evaluate these claims using a WideResNet model. For the WideResNet model, the specific quantitative claims are given in the following table:

Test error (%, flip/translation augmentation, mean/std normalization, mean of 5 runs) and “+” indicates standard data augmentation (mirror
+ crop)

| **Network** | **CIFAR-10** | **CIFAR-10+** | **CIFAR-100** | **CIFAR-100+** | **SVHN**
| ----------- | ------------ | ------------- | ------------ | ------------- |------------
| WideResNet    | 6.97         | 3.87      | 26.06         | 18.8      | 1.60
| WideResNet + cutout | 5.54   | 3.08         | 23.94        | 18.41       |  1.30


In this table, the effectiveness of standard and cutout data augmentation techniques is evaluated using the WideResNet architecture on the CIFAR-10, CIFAR-100, and SVHN datasets. The "+", as before, indicates the use of standard data augmentation (mirror and crop).

For CIFAR-10, utilizing the WideResNet model with standard augmentation significantly reduces the test error from 6.97% to 3.87%. When cutout augmentation is added, the error drops even further to 3.08%.

A comparable trend is seen with the CIFAR-100 dataset. Standard augmentation reduces the WideResNet model's test error from 26.06% to 18.8%. With the application of cutout augmentation, the error rate decreases slightly more to 18.41%.

Lastly, the SVHN dataset shows the smallest error rates. With standard augmentation, the error is 1.60% which further reduces to 1.30% with the addition of cutout augmentation.

These results demonstrate the robust effectiveness of both standard and cutout augmentation techniques in lowering test error rates across all tested datasets when used with the WideResNet model. As with the previous findings, the effect of augmentation appears to be influenced by the complexity of the dataset.

:::

::: {.cell .markdown}

## Import Library

:::


::: {.cell .code}
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm
import math
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
## 3.1 Implementation Code
:::

::: {.cell .markdown}
### 3.1.1 WideResNet Code
:::

::: {.cell .code}
``` python
# WideResNet

# From https://github.com/uoguelph-mlrg/Cutout/blob/master/model/wide_resnet.py


class BasicBlockWide(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlockWide, self).__init__()
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
        n = (depth - 4) // 6
        block = BasicBlockWide
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


::: {.cell .markdown}
### 3.1.2. Model Evaluate Test Code
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
## 3.2 Training WideResNet in CIFAR-10 
:::

::: {.cell .markdown}
### 3.2.1. Training WideResNet in CF10 without Cutout
::: 

::: {.cell .markdown}
Image Processing for CIFAR-10
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_cifar10 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform_cifar10 = transforms.Compose([])

train_transform_cifar10.transforms.append(transforms.ToTensor())
train_transform_cifar10.transforms.append(normalize_image_cifar10)



test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_cifar10])
```
:::

::: {.cell .markdown} 
Import the dataset of CIFAR-10 
:::

::: {.cell .code}
``` python
train_dataset_cifar10 = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform_cifar10,
                                     download=True)

test_dataset_cifar10 = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform_cifar10,
                                    download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_cifar10 = 128
train_loader_cifar10 = torch.utils.data.DataLoader(dataset=train_dataset_cifar10,
                                           batch_size=batch_size_cifar10,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_cifar10 = torch.utils.data.DataLoader(dataset=test_dataset_cifar10,
                                          batch_size=batch_size_cifar10,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_cifar10 = "wideresnet_cifar10"

num_classes_cifar10 = 10
wideresnet_cifar10 = WideResNet(depth=28, num_classes=num_classes_cifar10, widen_factor=10, dropRate=0.3)


wideresnet_cifar10 = wideresnet_cifar10.cuda()
learning_rate_wideresnet_cifar10 = 0.1
criterion_wideresnet_cifar10 = nn.CrossEntropyLoss().cuda()
cnn_optimizer_wideresnet_cifar10 = torch.optim.SGD(wideresnet_cifar10.parameters(), lr=learning_rate_wideresnet_cifar10,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_wideresnet_cifar10 = MultiStepLR(cnn_optimizer_wideresnet_cifar10, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown} 
Training WideResNet withuout Cutout 
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_cifar10)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_cifar10.zero_grad()
        pred = wideresnet_cifar10(images)

        xentropy_loss = criterion_wideresnet_cifar10(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_wideresnet_cifar10.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_accr_wideresnet_cifar10 = test(test_loader_cifar10, wideresnet_cifar10)
    tqdm.write('test_acc: %.3f' % (test_accr_wideresnet_cifar10))

    scheduler_wideresnet_cifar10.step()    

    
torch.save(wideresnet_cifar10.state_dict(), 'checkpoints/' + file_name_wideresnet_cifar10 + '.pt')


final_test_acc_wideresnet_cifar10 = (1 - test(test_loader_cifar10, wideresnet_cifar10))*100
print('Final Result WideResNet without Cutout for Test CIFAR-10 Dataset: %.3f' % (final_test_acc_wideresnet_cifar10))
``` 
:::


::: {.cell .markdown}
### 3.2.2. Training WideResNet in CF10 with Cutout
::: 

::: {.cell .markdown}
Cutout Code
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
Image Processing for CIFAR-10 
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_cifar10 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform_cifar10_cutout = transforms.Compose([])

train_transform_cifar10_cutout.transforms.append(transforms.ToTensor())
train_transform_cifar10_cutout.transforms.append(normalize_image_cifar10)

#Add Cutout to the image transformer pipeline
n_holes_cifar10 = 1
length_cifar10 = 16
train_transform_cifar10_cutout.transforms.append(Cutout(n_holes=n_holes_cifar10, length=length_cifar10))


test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_cifar10])
``` 
:::

::: {.cell .markdown} 
Import the dataset of CIFAR-10 
:::

::: {.cell .code}
``` python
train_dataset_cifar10_cutout = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform_cifar10_cutout,
                                     download=True)

test_dataset_cifar10 = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform_cifar10,
                                    download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_cifar10_cutout = 128
train_loader_cifar10_cutout = torch.utils.data.DataLoader(dataset=train_dataset_cifar10_cutout,
                                           batch_size=batch_size_cifar10_cutout,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_cifar10 = torch.utils.data.DataLoader(dataset=test_dataset_cifar10,
                                          batch_size=batch_size_cifar10_cutout,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_cifar10_cutout = "wideresnet_cifar10_cutout"

num_classes_cifar10 = 10
wideresnet_cifar10_cutout = WideResNet(depth=28, num_classes=num_classes_cifar10, widen_factor=10, dropRate=0.3)


wideresnet_cifar10_cutout = wideresnet_cifar10_cutout.cuda()
learning_rate_wideresnet_cifar10_cutout = 0.1
criterion_wideresnet_cifar10_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_wideresnet_cifar10_cutout = torch.optim.SGD(wideresnet_cifar10_cutout.parameters(), lr=learning_rate_wideresnet_cifar10_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_wideresnet_cifar10_cutout = MultiStepLR(cnn_optimizer_wideresnet_cifar10_cutout, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training WideResNet with Cutout 
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_cifar10_cutout)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_cifar10_cutout.zero_grad()
        pred = wideresnet_cifar10_cutout(images)

        xentropy_loss = criterion_wideresnet_cifar10_cutout(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_wideresnet_cifar10_cutout.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_cifar10 = test(test_loader_cifar10,wideresnet_cifar10_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_cifar10))
    scheduler_wideresnet_cifar10_cutout.step()     
torch.save(wideresnet_cifar10_cutout.state_dict(), 'checkpoints/' + file_name_wideresnet_cifar10_cutout + '.pt')


final_test_acc_wideresnet_cifar10_cutout = (1 - test(test_loader_cifar10,wideresnet_cifar10_cutout))*100
print('Final Result WideResNet using Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar10_cutout))
```
:::


::: {.cell .markdown}
### 3.2.3. Training WideResNet in CF10 with Data Augmentation 
::: 


::: {.cell .markdown} 
Image Processing for CIFAR-10 
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_cifar10 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform_cifar10_da = transforms.Compose([])
train_transform_cifar10_da.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform_cifar10_da.transforms.append(transforms.RandomHorizontalFlip())
train_transform_cifar10_da.transforms.append(transforms.ToTensor())
train_transform_cifar10_da.transforms.append(normalize_image_cifar10)


test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_cifar10])
``` 
:::

::: {.cell .markdown} 
Import the dataset of CIFAR-10 
:::

::: {.cell .code}
``` python
train_dataset_cifar10_da = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform_cifar10_da,
                                     download=True)

test_dataset_cifar10 = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform_cifar10,
                                    download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_cifar10_da = 128
train_loader_cifar10_da = torch.utils.data.DataLoader(dataset=train_dataset_cifar10_da,
                                           batch_size=batch_size_cifar10_da,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_cifar10 = torch.utils.data.DataLoader(dataset=test_dataset_cifar10,
                                          batch_size=batch_size_cifar10_da,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_cifar10_da = "wideresnet_cifar10_da"

num_classes_cifar10 = 10
wideresnet_cifar10_da = WideResNet(depth=28, num_classes=num_classes_cifar10, widen_factor=10, dropRate=0.3)


wideresnet_cifar10_da = wideresnet_cifar10_da.cuda()
learning_rate_wideresnet_cifar10_da = 0.1
criterion_wideresnet_cifar10_da = nn.CrossEntropyLoss().cuda()
cnn_optimizer_wideresnet_cifar10_da = torch.optim.SGD(wideresnet_cifar10_da.parameters(), lr=learning_rate_wideresnet_cifar10_da,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_wideresnet_cifar10_da = MultiStepLR(cnn_optimizer_wideresnet_cifar10_da, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training WideResNet with  Data Augmentation
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_cifar10_da)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_cifar10_da.zero_grad()
        pred = wideresnet_cifar10_da(images)

        xentropy_loss = criterion_wideresnet_cifar10_da(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_wideresnet_cifar10_da.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_wideresnet_cifar10_da = test(test_loader_cifar10,wideresnet_cifar10_da)
    tqdm.write('test_acc: %.3f' % (test_acc_wideresnet_cifar10_da))
    scheduler_wideresnet_cifar10_da.step()     
torch.save(wideresnet_cifar10_da.state_dict(), 'checkpoints/' + file_name_wideresnet_cifar10_da + '.pt')


final_test_acc_wideresnet_cifar10_da = (1 - test(test_loader_cifar10,wideresnet_cifar10_da))*100
print('Final Result WideResNet using Data Augmentation for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar10_da))
```
:::


::: {.cell .markdown}
### 3.2.4. Training WideResNet in CF10 with Data Augmentation with Cutout
::: 

::: {.cell .markdown} 
Image Processing for CIFAR-10 
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_cifar10 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform_cifar10_da_co = transforms.Compose([])
train_transform_cifar10_da_co.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform_cifar10_da_co.transforms.append(transforms.RandomHorizontalFlip())
train_transform_cifar10_da_co.transforms.append(transforms.ToTensor())
train_transform_cifar10_da_co.transforms.append(normalize_image_cifar10)

#Add Cutout to the image transformer pipeline
n_holes_cifar10_da_co = 1
length_cifar10_da_co = 16
train_transform_cifar10_da_co.transforms.append(Cutout(n_holes=n_holes_cifar10_da_co, length=length_cifar10_da_co))


test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_cifar10])
``` 
:::

::: {.cell .markdown} 
Import the dataset of CIFAR-10 
:::

::: {.cell .code}
``` python
train_dataset_cifar10_da_co = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform_cifar10_da_co,
                                     download=True)

test_dataset_cifar10 = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform_cifar10,
                                    download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_cifar10_da_co = 128
train_loader_cifar10_da_co = torch.utils.data.DataLoader(dataset=train_dataset_cifar10_da_co,
                                           batch_size=batch_size_cifar10_da_co,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_cifar10 = torch.utils.data.DataLoader(dataset=test_dataset_cifar10,
                                          batch_size=batch_size_cifar10_da_co,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_cifar10_da_cutout = "wideresnet_cifar10_da_cutout"

num_classes_cifar10 = 10
wideresnet_cifar10_da_cutout = WideResNet(depth=28, num_classes=num_classes_cifar10, widen_factor=10, dropRate=0.3)


wideresnet_cifar10_da_cutout = wideresnet_cifar10_da_cutout.cuda()
learning_rate_cifar10_da_cutout = 0.1
criterion_cifar10_da_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_cifar10_da_cutout = torch.optim.SGD(wideresnet_cifar10_da_cutout.parameters(), lr=learning_rate_cifar10_da_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_cifar10_da_cutout = MultiStepLR(cnn_optimizer_cifar10_da_cutout, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training WideResNet with Cutout 
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_cifar10_da_co)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_cifar10_da_cutout.zero_grad()
        pred = wideresnet_cifar10_da_cutout(images)

        xentropy_loss = criterion_cifar10_da_cutout(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_cifar10_da_cutout.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_cifar10_da_cutout = test(test_loader_cifar10,wideresnet_cifar10_da_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_cifar10_da_cutout))
    scheduler_cifar10_da_cutout.step()     
torch.save(wideresnet_cifar10_da_cutout.state_dict(), 'checkpoints/' + file_name_wideresnet_cifar10_da_cutout + '.pt')


final_test_acc_wideresnet_cifar10_da_cutout = (1 - test(test_loader_cifar10,wideresnet_cifar10_da_cutout))*100
print('Final Result WideResNet using Data Augmentation and  Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar10_da_cutout))
```
:::

::: {.cell .code}
``` python
print('Final Result WideResNet without Cutout for Test CIFAR-10 Dataset: %.3f' % (final_test_acc_wideresnet_cifar10))
print('Final Result WideResNet using Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar10_cutout))
print('Final Result WideResNet using Data Augmentation for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar10_da))
print('Final Result WideResNet using Data Augmentation and  Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar10_da_cutout))
```
:::

::: {.cell .markdown}
## 3.3 Training WideResNet in CIFAR-100 
:::

::: {.cell .markdown}
### 3.3.1. Training WideResNet in CF100 without Cutout
::: 

::: {.cell .markdown}
Image Processing for CIFAR-100
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_cifar100 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform_cifar100 = transforms.Compose([])

train_transform_cifar100.transforms.append(transforms.ToTensor())
train_transform_cifar100.transforms.append(normalize_image_cifar100)



test_transform_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_cifar100])
```
:::

::: {.cell .markdown} 
Import the dataset of CIFAR-100 
:::

::: {.cell .code}
``` python
train_dataset_cifar100 = datasets.CIFAR100(root='data/',
                                     train=True,
                                     transform=train_transform_cifar100,
                                     download=True)

test_dataset_cifar100 = datasets.CIFAR100(root='data/',
                                    train=False,
                                    transform=test_transform_cifar100,
                                    download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_cifar100 = 128
train_loader_cifar100 = torch.utils.data.DataLoader(dataset=train_dataset_cifar100,
                                           batch_size=batch_size_cifar100,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_cifar100 = torch.utils.data.DataLoader(dataset=test_dataset_cifar100,
                                          batch_size=batch_size_cifar100,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_cifar100 = "wideresnet_cifar100"

num_classes_cifar100 = 100
wideresnet_cifar100 = WideResNet(depth=28, num_classes=num_classes_cifar100, widen_factor=10, dropRate=0.3)


wideresnet_cifar100 = wideresnet_cifar100.cuda()
learning_rate_wideresnet_cifar100 = 0.1
criterion_wideresnet_cifar100 = nn.CrossEntropyLoss().cuda()
cnn_optimizer_wideresnet_cifar100 = torch.optim.SGD(wideresnet_cifar100.parameters(), lr=learning_rate_wideresnet_cifar100,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_wideresnet_cifar100 = MultiStepLR(cnn_optimizer_wideresnet_cifar100, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown} 
Training WideResNet withuout Cutout 
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_cifar100)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_cifar100.zero_grad()
        pred = wideresnet_cifar100(images)

        xentropy_loss = criterion_wideresnet_cifar100(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_wideresnet_cifar100.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_accr_wideresnet_cifar100 = test(test_loader_cifar100, wideresnet_cifar100)
    tqdm.write('test_acc: %.3f' % (test_accr_wideresnet_cifar100))

    scheduler_wideresnet_cifar100.step()    

    
torch.save(wideresnet_cifar100.state_dict(), 'checkpoints/' + file_name_wideresnet_cifar100 + '.pt')


final_test_acc_wideresnet_cifar100 = (1 - test(test_loader_cifar100, wideresnet_cifar100))*100
print('Final Result WideResNet without Cutout for Test CIFAR-100 Dataset: %.3f' % (final_test_acc_wideresnet_cifar100))
``` 
:::


::: {.cell .markdown}
### 3.2.2. Training WideResNet in CF100 with Cutout
::: 



::: {.cell .markdown} 
Image Processing for CIFAR-100 
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_cifar100 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform_cifar100_cutout = transforms.Compose([])

train_transform_cifar100_cutout.transforms.append(transforms.ToTensor())
train_transform_cifar100_cutout.transforms.append(normalize_image_cifar100)

#Add Cutout to the image transformer pipeline
n_holes_cifar100 = 1
length_cifar100 = 8
train_transform_cifar100_cutout.transforms.append(Cutout(n_holes=n_holes_cifar100, length=length_cifar100))


test_transform_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_cifar100])
``` 
:::

::: {.cell .markdown} 
Import the dataset of CIFAR-0 
:::

::: {.cell .code}
``` python
train_dataset_cifar100_cutout = datasets.CIFAR100(root='data/',
                                     train=True,
                                     transform=train_transform_cifar100_cutout,
                                     download=True)

test_dataset_cifar100 = datasets.CIFAR100(root='data/',
                                    train=False,
                                    transform=test_transform_cifar100,
                                    download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_cifar100_cutout = 128
train_loader_cifar100_cutout = torch.utils.data.DataLoader(dataset=train_dataset_cifar100_cutout,
                                           batch_size=batch_size_cifar100_cutout,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_cifar100 = torch.utils.data.DataLoader(dataset=test_dataset_cifar100,
                                          batch_size=batch_size_cifar100_cutout,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_cifar100_cutout = "wideresnet_cifar100_cutout"

num_classes_cifar100 = 100
wideresnet_cifar100_cutout = WideResNet(depth=28, num_classes=num_classes_cifar100, widen_factor=10, dropRate=0.3)


wideresnet_cifar100_cutout = wideresnet_cifar100_cutout.cuda()
learning_rate_wideresnet_cifar100_cutout = 0.1
criterion_wideresnet_cifar100_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_wideresnet_cifar100_cutout = torch.optim.SGD(wideresnet_cifar100_cutout.parameters(), lr=learning_rate_wideresnet_cifar100_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_wideresnet_cifar100_cutout = MultiStepLR(cnn_optimizer_wideresnet_cifar100_cutout, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training WideResNet with Cutout 
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_cifar100_cutout)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_cifar100_cutout.zero_grad()
        pred = wideresnet_cifar100_cutout(images)

        xentropy_loss = criterion_wideresnet_cifar100_cutout(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_wideresnet_cifar100_cutout.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_cifar100 = test(test_loader_cifar100,wideresnet_cifar100_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_cifar100))
    scheduler_wideresnet_cifar100_cutout.step()     
torch.save(wideresnet_cifar100_cutout.state_dict(), 'checkpoints/' + file_name_wideresnet_cifar100_cutout + '.pt')


final_test_acc_wideresnet_cifar100_cutout = (1 - test(test_loader_cifar100,wideresnet_cifar100_cutout))*100
print('Final Result WideResNet using Cutout for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar100_cutout))
```
:::


::: {.cell .markdown}
### 3.2.3. Training WideResNet in CF100 with Data Augmentation 
::: 


::: {.cell .markdown} 
Image Processing for CIFAR-100 
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_cifar100 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform_cifar100_da = transforms.Compose([])
train_transform_cifar100_da.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform_cifar100_da.transforms.append(transforms.RandomHorizontalFlip())
train_transform_cifar100_da.transforms.append(transforms.ToTensor())
train_transform_cifar100_da.transforms.append(normalize_image_cifar100)


test_transform_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_cifar100])
``` 
:::

::: {.cell .markdown} 
Import the dataset of CIFAR-100
:::

::: {.cell .code}
``` python
train_dataset_cifar100_da = datasets.CIFAR100(root='data/',
                                     train=True,
                                     transform=train_transform_cifar100_da,
                                     download=True)

test_dataset_cifar100 = datasets.CIFAR100(root='data/',
                                    train=False,
                                    transform=test_transform_cifar100,
                                    download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_cifar100_da = 128
train_loader_cifar100_da = torch.utils.data.DataLoader(dataset=train_dataset_cifar100_da,
                                           batch_size=batch_size_cifar100_da,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_cifar100 = torch.utils.data.DataLoader(dataset=test_dataset_cifar100,
                                          batch_size=batch_size_cifar100_da,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_cifar100_da = "wideresnet_cifar100_da"

num_classes_cifar100 = 100
wideresnet_cifar100_da = WideResNet(depth=28, num_classes=num_classes_cifar100, widen_factor=10, dropRate=0.3)



wideresnet_cifar100_da = wideresnet_cifar100_da.cuda()
learning_rate_wideresnet_cifar100_da = 0.1
criterion_wideresnet_cifar100_da = nn.CrossEntropyLoss().cuda()
cnn_optimizer_wideresnet_cifar100_da = torch.optim.SGD(wideresnet_cifar100_da.parameters(), lr=learning_rate_wideresnet_cifar100_da,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_wideresnet_cifar100_da = MultiStepLR(cnn_optimizer_wideresnet_cifar100_da, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training WideResNet with  Data Augmentation
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_cifar100_da)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_cifar100_da.zero_grad()
        pred = wideresnet_cifar100_da(images)

        xentropy_loss = criterion_wideresnet_cifar100_da(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_wideresnet_cifar100_da.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_wideresnet_cifar100_da = test(test_loader_cifar100,wideresnet_cifar100_da)
    tqdm.write('test_acc: %.3f' % (test_acc_wideresnet_cifar100_da))
    scheduler_wideresnet_cifar100_da.step()     
torch.save(wideresnet_cifar100_da.state_dict(), 'checkpoints/' + file_name_wideresnet_cifar100_da + '.pt')


final_test_acc_wideresnet_cifar100_da = (1 - test(test_loader_cifar100,wideresnet_cifar100_da))*100
print('Final Result WideResNet using Data Augmentation for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar100_da))
```
:::


::: {.cell .markdown}
### 3.2.4. Training WideResNet in CF100 with Data Augmentation with Cutout
::: 

::: {.cell .markdown} 
Image Processing for CIFAR-100
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_cifar100 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform_cifar100_da_co = transforms.Compose([])
train_transform_cifar100_da_co.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform_cifar100_da_co.transforms.append(transforms.RandomHorizontalFlip())
train_transform_cifar100_da_co.transforms.append(transforms.ToTensor())
train_transform_cifar100_da_co.transforms.append(normalize_image_cifar100)

#Add Cutout to the image transformer pipeline
n_holes_cifar100_da_co = 1
length_cifar100_da_co = 8
train_transform_cifar100_da_co.transforms.append(Cutout(n_holes=n_holes_cifar100_da_co, length=length_cifar100_da_co))


test_transform_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_cifar100])
``` 
:::

::: {.cell .markdown} 
Import the dataset of CIFAR-100
:::

::: {.cell .code}
``` python
train_dataset_cifar100_da_co = datasets.CIFAR100(root='data/',
                                     train=True,
                                     transform=train_transform_cifar100_da_co,
                                     download=True)

test_dataset_cifar100 = datasets.CIFAR100(root='data/',
                                    train=False,
                                    transform=test_transform_cifar100,
                                    download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_cifar100_da_co = 128
train_loader_cifar100_da_co = torch.utils.data.DataLoader(dataset=train_dataset_cifar100_da_co,
                                           batch_size=batch_size_cifar100_da_co,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_cifar100 = torch.utils.data.DataLoader(dataset=test_dataset_cifar100,
                                          batch_size=batch_size_cifar100_da_co,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_cifar100_da_cutout = "wideresnet_cifar100_da_cutout"

num_classes_cifar100 = 100
wideresnet_cifar100_da_cutout = WideResNet(depth=28, num_classes=num_classes_cifar100, widen_factor=10, dropRate=0.3)


wideresnet_cifar100_da_cutout = wideresnet_cifar100_da_cutout.cuda()
learning_rate_cifar100_da_cutout = 0.1
criterion_cifar100_da_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_cifar100_da_cutout = torch.optim.SGD(wideresnet_cifar100_da_cutout.parameters(), lr=learning_rate_cifar100_da_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_cifar100_da_cutout = MultiStepLR(cnn_optimizer_cifar100_da_cutout, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training WideResNet with Cutout 
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_cifar100_da_co)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_cifar100_da_cutout.zero_grad()
        pred = wideresnet_cifar100_da_cutout(images)

        xentropy_loss = criterion_cifar100_da_cutout(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_cifar100_da_cutout.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_cifar100_da_cutout = test(test_loader_cifar100,wideresnet_cifar100_da_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_cifar100_da_cutout))
    scheduler_cifar100_da_cutout.step()     
torch.save(wideresnet_cifar100_da_cutout.state_dict(), 'checkpoints/' + file_name_wideresnet_cifar100_da_cutout + '.pt')


final_test_acc_wideresnet_cifar100_da_cutout = (1 - test(test_loader_cifar100,wideresnet_cifar100_da_cutout))*100
print('Final Result WideResNet using Data Augmentation and  Cutout for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar100_da_cutout))
```
:::

::: {.cell .code}
``` python
print('Final Result WideResNet without Cutout for Test CIFAR-100 Dataset: %.3f' % (final_test_acc_wideresnet_cifar100))
print('Final Result WideResNet using Cutout for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar100_cutout))
print('Final Result WideResNet using Data Augmentation for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar100_da))
print('Final Result WideResNet using Data Augmentation and  Cutout for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_wideresnet_cifar100_da_cutout))
```
:::


::: {.cell .markdown}
## 3.4 Training WideResNet in SVHN
:::

::: {.cell .markdown}
### 3.4.1. Training WideResNet in SVHN without Cutout
::: 

::: {.cell .markdown}
Image Processing for SVHN
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_svhn = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],std=[x / 255.0 for x in [50.1, 50.6, 50.8]])

train_transform_svhn = transforms.Compose([])

train_transform_svhn.transforms.append(transforms.ToTensor())
train_transform_svhn.transforms.append(normalize_image_svhn)



test_transform_svhn = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_svhn])
```
:::

::: {.cell .markdown} 
Import the dataset of SVHN
:::

::: {.cell .code}
``` python
train_dataset_svhn = datasets.SVHN(root='data/',
                                    split='train',
                                    transform=train_transform_svhn,
                                    download=True)

extra_dataset_svhn = datasets.SVHN(root='data/',
                                    split='extra',
                                    transform=train_transform_svhn,
                                    download=True)

# Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
data_svhn = np.concatenate([train_dataset_svhn.data, extra_dataset_svhn.data], axis=0)
labels_svhn = np.concatenate([train_dataset_svhn.labels, extra_dataset_svhn.labels], axis=0)
train_dataset_svhn.data = data_svhn
train_dataset_svhn.labels = labels_svhn

test_dataset_svhn = datasets.SVHN(root='data/',
                                  split='test',
                                  transform=test_transform_svhn,
                                  download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_svhn = 128
train_loader_svhn = torch.utils.data.DataLoader(dataset=train_dataset_svhn,
                                           batch_size=batch_size_svhn,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_svhn = torch.utils.data.DataLoader(dataset=test_dataset_svhn,
                                          batch_size=batch_size_svhn,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_svhn = "wideresnet_svhn"

num_classes_svhn = 10
wideresnet_svhn = WideResNet(depth=16, num_classes=num_classes_svhn, widen_factor=8,dropRate=0.4)


wideresnet_svhn = wideresnet_svhn.cuda()
learning_rate_wideresnet_svhn = 0.01
criterion_wideresnet_svhn = nn.CrossEntropyLoss().cuda()
cnn_optimizer_wideresnet_svhn = torch.optim.SGD(wideresnet_svhn.parameters(), lr=learning_rate_wideresnet_svhn,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_wideresnet_svhn = MultiStepLR(cnn_optimizer_wideresnet_svhn, milestones=[80, 120], gamma=0.1)
```
:::

::: {.cell .markdown} 
Training WideResNet withuout Cutout 
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 160
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_svhn)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_svhn.zero_grad()
        pred = wideresnet_svhn(images)

        xentropy_loss = criterion_wideresnet_svhn(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_wideresnet_svhn.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_accr_wideresnet_svhn = test(test_loader_svhn, wideresnet_svhn)
    tqdm.write('test_acc: %.3f' % (test_accr_wideresnet_svhn))

    scheduler_wideresnet_svhn.step()     

    
torch.save(wideresnet_svhn.state_dict(), 'checkpoints/' + file_name_wideresnet_svhn + '.pt')


final_test_acc_wideresnet_svhn = (1 - test(test_loader_svhn, wideresnet_svhn))*100
print('Final Result WideResNet without Cutout for Test SVHN Dataset: %.3f' % (final_test_acc_wideresnet_svhn))
``` 
:::


::: {.cell .markdown}
### 3.4.2. Training WideResNet in SVHN with Cutout
::: 

::: {.cell .markdown} 
Image Processing for SVHN
:::

::: {.cell .code}
``` python
# Image Preprocessing

normalize_image_svhn = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]], std=[x / 255.0 for x in [50.1, 50.6, 50.8]])

train_transform_svhn_cutout = transforms.Compose([])

train_transform_svhn_cutout.transforms.append(transforms.ToTensor())
train_transform_svhn_cutout.transforms.append(normalize_image_svhn)

#Add Cutout to the image transformer pipeline
n_holes_svhn = 1
length_svhn = 20
train_transform_svhn_cutout.transforms.append(Cutout(n_holes=n_holes_svhn, length=length_svhn))


test_transform_svhn = transforms.Compose([
    transforms.ToTensor(),
    normalize_image_svhn])
``` 
:::

::: {.cell .markdown} 
Import the dataset of SVHN
:::

::: {.cell .code}
``` python
train_dataset_svhn_cutout = datasets.SVHN(root='data/',
                                    split='train',
                                    transform=train_transform_svhn_cutout,
                                    download=True)

extra_dataset_svhn_cutout = datasets.SVHN(root='data/',
                                    split='extra',
                                    transform=train_transform_svhn_cutout,
                                    download=True)

# Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
data_svhn_cutout = np.concatenate([train_dataset_svhn_cutout.data, extra_dataset_svhn_cutout.data], axis=0)
labels_svhn_cutout = np.concatenate([train_dataset_svhn_cutout.labels, extra_dataset_svhn_cutout.labels], axis=0)
train_dataset_svhn_cutout.data = data_svhn_cutout
train_dataset_svhn_cutout.labels = labels_svhn_cutout

test_dataset_svhn = datasets.SVHN(root='data/',
                                  split='test',
                                  transform=test_transform_svhn,
                                  download=True)
```
:::

::: {.cell .markdown} 
Create Dataset as Dataloader 
:::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size_svhn_cutout = 128
train_loader_svhn_cutout = torch.utils.data.DataLoader(dataset=train_dataset_svhn_cutout,
                                           batch_size=batch_size_svhn_cutout,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader_svhn = torch.utils.data.DataLoader(dataset=test_dataset_svhn,
                                          batch_size=batch_size_svhn_cutout,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} 
Define the model 
:::

::: {.cell .markdown} 
This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. 
:::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name_wideresnet_svhn_cutout = "wideresnet_svhn_cutout"

num_classes_svhn = 10
wideresnet_svhn_cutout = WideResNet(depth=16, num_classes=num_classes_svhn, widen_factor=8,dropRate=0.4)


wideresnet_svhn_cutout = wideresnet_svhn_cutout.cuda()
learning_rate_wideresnet_svhn_cutout = 0.01
criterion_wideresnet_svhn_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_wideresnet_svhn_cutout = torch.optim.SGD(wideresnet_svhn_cutout.parameters(), lr=learning_rate_wideresnet_svhn_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_wideresnet_svhn_cutout = MultiStepLR(cnn_optimizer_wideresnet_svhn_cutout, milestones=[80, 120], gamma=0.1)
```
:::

::: {.cell .markdown}
Training WideResNet with Cutout 
:::

::: {.cell .markdown} 
This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. 
:::

::: {.cell .code}
``` python
epochs = 160
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader_svhn_cutout)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        wideresnet_svhn_cutout.zero_grad()
        pred = wideresnet_svhn_cutout(images)

        xentropy_loss = criterion_wideresnet_svhn_cutout(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_wideresnet_svhn_cutout.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_svhn = test(test_loader_svhn,wideresnet_svhn_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_svhn))
    scheduler_wideresnet_svhn_cutout.step()     
torch.save(wideresnet_svhn_cutout.state_dict(), 'checkpoints/' + file_name_wideresnet_svhn_cutout + '.pt')


final_test_acc_wideresnet_svhn_cutout = (1 - test(test_loader_svhn,wideresnet_svhn_cutout))*100
print('Final Result WideResNet using Cutout for SVHN Test Dataset: %.3f' % (final_test_acc_wideresnet_svhn_cutout))
```
:::

::: {.cell .code}
``` python
print('Final Result WideResNet without Cutout for Test SVHN Dataset: %.3f' % (final_test_acc_wideresnet_svhn))
print('Final Result WideResNet using Cutout for SVHN Test Dataset: %.3f' % (final_test_acc_wideresnet_svhn_cutout))
```
:::
