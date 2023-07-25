
::: {.cell .markdown}
### 02. ResNet
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

### Import Library

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
### 2.1.1 ResNet Code
:::

::: {.cell .code}
``` python
# ResNet
# From https://github.com/uoguelph-mlrg/Cutout/blob/master/model/resnet.py

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
```
:::


::: {.cell .markdown}
### 2.1.2. Model Evaluate Test Code
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
### 2.2.1. Training ResNet-18 in CF10 without Cutout
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
file_name_resnet18_cifar10 = "resnet18_cifar10"

num_classes_cifar10 = 10
resnet18_cifar10 = ResNet18(num_classes=num_classes_cifar10)


resnet18_cifar10 = resnet18_cifar10.cuda()
learning_rate_resnet18_cifar10 = 0.1
criterion_resnet18_cifar10 = nn.CrossEntropyLoss().cuda()
cnn_optimizer_resnet18_cifar10 = torch.optim.SGD(resnet18_cifar10.parameters(), lr=learning_rate_resnet18_cifar10,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_resnet18_cifar10 = MultiStepLR(cnn_optimizer_resnet18_cifar10, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown} 
Training ResNet-18 withuout Cutout 
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

        resnet18_cifar10.zero_grad()
        pred = resnet18_cifar10(images)

        xentropy_loss = criterion_resnet18_cifar10(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_resnet18_cifar10.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_accr_resnet18_cifar10 = test(test_loader_cifar10, resnet18_cifar10)
    tqdm.write('test_acc: %.3f' % (test_accr_resnet18_cifar10))

    #scheduler.step(epoch)  # Use this line for PyTorch <1.4
    scheduler_resnet18_cifar10.step()     # Use this line for PyTorch >=1.4

    
torch.save(resnet18_cifar10.state_dict(), 'checkpoints/' + file_name_resnet18_cifar10 + '.pt')


final_test_acc_resnet18_cifar10 = (1 - test(test_loader_cifar10, resnet18_cifar10))*100
print('Final Result ResNet-18 without Cutout for Test CIFAR-10 Dataset: %.3f' % (final_test_acc_resnet18_cifar10))
``` 
:::


::: {.cell .markdown}
### 2.2.2. Training ResNet-18 in CF10 with Cutout
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

#Add Cutout to the image transformer piepeline
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
file_name_resnet18_cifar10_cutout = "resnet18_cifar10_cutout"

num_classes_cifar10 = 10
resnet18_cifar10_cutout = ResNet18(num_classes=num_classes_cifar10)


resnet18_cifar10_cutout = resnet18_cifar10_cutout.cuda()
learning_rate_resnet18_cifar10_cutout = 0.1
criterion_resnet18_cifar10_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_resnet18_cifar10_cutout = torch.optim.SGD(resnet18_cifar10_cutout.parameters(), lr=learning_rate_resnet18_cifar10_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_resnet18_cifar10_cutout = MultiStepLR(cnn_optimizer_resnet18_cifar10_cutout, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training ResNet-18 with Cutout 
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

        resnet18_cifar10_cutout.zero_grad()
        pred = resnet18_cifar10_cutout(images)

        xentropy_loss = criterion_resnet18_cifar10_cutout(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_resnet18_cifar10_cutout.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_cifar10 = test(test_loader_cifar10,resnet18_cifar10_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_cifar10))
    scheduler_resnet18_cifar10_cutout.step()     
torch.save(resnet18_cifar10_cutout.state_dict(), 'checkpoints/' + file_name_resnet18_cifar10_cutout + '.pt')


final_test_acc_resnet18_cifar10_cutout = (1 - test(test_loader_cifar10_cutout,resnet18_cifar10_cutout))*100
print('Final Result ResNet-18 using Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar10_cutout))
```
:::


::: {.cell .markdown}
### 2.2.3. Training ResNet-18 in CF10 with Data Augmentation 
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
file_name_resnet18_cifar10_da = "resnet18_cifar10_da"

num_classes_cifar10 = 10
resnet18_cifar10_da = ResNet18(num_classes=num_classes_cifar10)


resnet18_cifar10_da = resnet18_cifar10_da.cuda()
learning_rate_resnet18_cifar10_da = 0.1
criterion_resnet18_cifar10_da = nn.CrossEntropyLoss().cuda()
cnn_optimizer_resnet18_cifar10_da = torch.optim.SGD(resnet18_cifar10_da.parameters(), lr=learning_rate_resnet18_cifar10_da,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_resnet18_cifar10_da = MultiStepLR(cnn_optimizer_resnet18_cifar10_da, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training ResNet-18 with  Data Augmentation
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

        resnet18_cifar10_da.zero_grad()
        pred = resnet18_cifar10_da(images)

        xentropy_loss = criterion_resnet18_cifar10_da(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_resnet18_cifar10_da.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_resnet18_cifar10_da = test(test_loader_cifar10,resnet18_cifar10_da)
    tqdm.write('test_acc: %.3f' % (test_acc_resnet18_cifar10_da))
    scheduler_resnet18_cifar10_da.step()     
torch.save(resnet18_cifar10_da.state_dict(), 'checkpoints/' + file_name_resnet18_cifar10_da + '.pt')


final_test_acc_resnet18_cifar10_da = (1 - test(test_loader_cifar10,resnet18_cifar10_da))*100
print('Final Result ResNet-18 using Data Augmentation for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar10_da))
```
:::


::: {.cell .markdown}
### 2.2.4. Training ResNet-18 in CF10 with Data Augmentation with Cutout
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

#Add Cutout to the image transformer piepeline
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
file_name_resnet18_cifar10_da_cutout = "resnet18_cifar10_da_cutout"

num_classes_cifar10 = 10
resnet18_cifar10_da_cutout = ResNet18(num_classes=num_classes_cifar10)


resnet18_cifar10_da_cutout = resnet18_cifar10_da_cutout.cuda()
learning_rate_cifar10_da_cutout = 0.1
criterion_cifar10_da_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_cifar10_da_cutout = torch.optim.SGD(resnet18_cifar10_da_cutout.parameters(), lr=learning_rate_cifar10_da_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_cifar10_da_cutout = MultiStepLR(cnn_optimizer_cifar10_da_cutout, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training ResNet-18 with Cutout 
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

        resnet18_cifar10_da_cutout.zero_grad()
        pred = resnet18_cifar10_da_cutout(images)

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

    test_acc_cifar10_da_cutout = test(test_loader_cifar10,resnet18_cifar10_da_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_cifar10_da_cutout))
    scheduler_cifar10_da_cutout.step()     
torch.save(resnet18_cifar10_da_cutout.state_dict(), 'checkpoints/' + file_name_resnet18_cifar10_da_cutout + '.pt')


final_test_acc_resnet18_cifar10_da_cutout = (1 - test(test_loader_cifar10,resnet18_cifar10_da_cutout))*100
print('Final Result ResNet-18 using Data Augmentation and  Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar10_da_cutout))
```
:::

::: {.cell .code}
``` python
print('Final Result ResNet-18 without Cutout for Test CIFAR-10 Dataset: %.3f' % (final_test_acc_resnet18_cifar10))
print('Final Result ResNet-18 using Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar10_cutout))
print('Final Result ResNet-18 using Data Augmentation for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar10_da))
print('Final Result ResNet-18 using Data Augmentation and  Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_resnet_cifar10_da_cutout))
```
:::


::: {.cell .markdown}
### 2.3.1. Training ResNet-18 in CF100 without Cutout
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
train_dataset_cifar100 = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform_cifar100,
                                     download=True)

test_dataset_cifar100 = datasets.CIFAR10(root='data/',
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
file_name_resnet18_cifar100 = "resnet18_cifar100"

num_classes_cifar100 = 100
resnet18_cifar100 = ResNet18(num_classes=num_classes_cifar100)


resnet18_cifar100 = resnet18_cifar100.cuda()
learning_rate_resnet18_cifar100 = 0.1
criterion_resnet18_cifar100 = nn.CrossEntropyLoss().cuda()
cnn_optimizer_resnet18_cifar100 = torch.optim.SGD(resnet18_cifar100.parameters(), lr=learning_rate_resnet18_cifar100,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_resnet18_cifar100 = MultiStepLR(cnn_optimizer_resnet18_cifar100, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown} 
Training ResNet-18 withuout Cutout 
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

        resnet18_cifar100.zero_grad()
        pred = resnet18_cifar100(images)

        xentropy_loss = criterion_resnet18_cifar100(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_resnet18_cifar100.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_accr_resnet18_cifar100 = test(test_loader_cifar100, resnet18_cifar100)
    tqdm.write('test_acc: %.3f' % (test_accr_resnet18_cifar100))

    #scheduler.step(epoch)  # Use this line for PyTorch <1.4
    scheduler_resnet18_cifar100.step()     # Use this line for PyTorch >=1.4

    
torch.save(resnet18_cifar100.state_dict(), 'checkpoints/' + file_name_resnet18_cifar100 + '.pt')


final_test_acc_resnet18_cifar100 = (1 - test(test_loader_cifar100, resnet18_cifar100))*100
print('Final Result ResNet-18 without Cutout for Test CIFAR-100 Dataset: %.3f' % (final_test_acc_resnet18_cifar100))
``` 
:::


::: {.cell .markdown}
### 2.2.2. Training ResNet-18 in CF100 with Cutout
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

#Add Cutout to the image transformer piepeline
n_holes_cifar100 = 1
length_cifar100 = 16
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
file_name_resnet18_cifar100_cutout = "resnet18_cifar100_cutout"

num_classes_cifar100 = 100
resnet18_cifar100_cutout = ResNet18(num_classes=num_classes_cifar100)


resnet18_cifar100_cutout = resnet18_cifar100_cutout.cuda()
learning_rate_resnet18_cifar100_cutout = 0.1
criterion_resnet18_cifar100_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_resnet18_cifar100_cutout = torch.optim.SGD(resnet18_cifar100_cutout.parameters(), lr=learning_rate_resnet18_cifar100_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_resnet18_cifar100_cutout = MultiStepLR(cnn_optimizer_resnet18_cifar100_cutout, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training ResNet-18 with Cutout 
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

        resnet18_cifar100_cutout.zero_grad()
        pred = resnet18_cifar100_cutout(images)

        xentropy_loss = criterion_resnet18_cifar100_cutout(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_resnet18_cifar100_cutout.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_cifar100 = test(test_loader_cifar100,resnet18_cifar100_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_cifar100))
    scheduler_resnet18_cifar100_cutout.step()     
torch.save(resnet18_cifar100_cutout.state_dict(), 'checkpoints/' + file_name_resnet18_cifar100_cutout + '.pt')


final_test_acc_resnet18_cifar100_cutout = (1 - test(test_loader_cifar100_cutout,resnet18_cifar100_cutout))*100
print('Final Result ResNet-18 using Cutout for CIFAR-10 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar100_cutout))
```
:::


::: {.cell .markdown}
### 2.2.3. Training ResNet-18 in CF100 with Data Augmentation 
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
file_name_resnet18_cifar100_da = "resnet18_cifar100_da"

num_classes_cifar100 = 100
resnet18_cifar100_da = ResNet18(num_classes=num_classes_cifar100)


resnet18_cifar100_da = resnet18_cifar100_da.cuda()
learning_rate_resnet18_cifar100_da = 0.1
criterion_resnet18_cifar100_da = nn.CrossEntropyLoss().cuda()
cnn_optimizer_resnet18_cifar100_da = torch.optim.SGD(resnet18_cifar100_da.parameters(), lr=learning_rate_resnet18_cifar100_da,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_resnet18_cifar100_da = MultiStepLR(cnn_optimizer_resnet18_cifar100_da, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training ResNet-18 with  Data Augmentation
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

        resnet18_cifar100_da.zero_grad()
        pred = resnet18_cifar100_da(images)

        xentropy_loss = criterion_resnet18_cifar100_da(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer_resnet18_cifar100_da.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc_resnet18_cifar100_da = test(test_loader_cifar100,resnet18_cifar100_da)
    tqdm.write('test_acc: %.3f' % (test_acc_resnet18_cifar100_da))
    scheduler_resnet18_cifar100_da.step()     
torch.save(resnet18_cifar100_da.state_dict(), 'checkpoints/' + file_name_resnet18_cifar100_da + '.pt')


final_test_acc_resnet18_cifar100_da = (1 - test(test_loader_cifar100,resnet18_cifar100_da))*100
print('Final Result ResNet-18 using Data Augmentation for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar100_da))
```
:::


::: {.cell .markdown}
### 2.2.4. Training ResNet-18 in CF100 with Data Augmentation with Cutout
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

#Add Cutout to the image transformer piepeline
n_holes_cifar100_da_co = 1
length_cifar100_da_co = 16
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
file_name_resnet18_cifar100_da_cutout = "resnet18_cifar100_da_cutout"

num_classes_cifar100 = 100
resnet18_cifar100_da_cutout = ResNet18(num_classes=num_classes_cifar100)


resnet18_cifar100_da_cutout = resnet18_cifar100_da_cutout.cuda()
learning_rate_cifar100_da_cutout = 0.1
criterion_cifar100_da_cutout = nn.CrossEntropyLoss().cuda()
cnn_optimizer_cifar100_da_cutout = torch.optim.SGD(resnet18_cifar100_da_cutout.parameters(), lr=learning_rate_cifar100_da_cutout,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler_cifar100_da_cutout = MultiStepLR(cnn_optimizer_cifar100_da_cutout, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown}
Training ResNet-18 with Cutout 
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

        resnet18_cifar100_da_cutout.zero_grad()
        pred = resnet18_cifar100_da_cutout(images)

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

    test_acc_cifar100_da_cutout = test(test_loader_cifar100,resnet18_cifar100_da_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc_cifar100_da_cutout))
    scheduler_cifar100_da_cutout.step()     
torch.save(resnet18_cifar100_da_cutout.state_dict(), 'checkpoints/' + file_name_resnet18_cifar100_da_cutout + '.pt')


final_test_acc_resnet18_cifar100_da_cutout = (1 - test(test_loader_cifar100,resnet18_cifar100_da_cutout))*100
print('Final Result ResNet-18 using Data Augmentation and  Cutout for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar100_da_cutout))
```
:::

::: {.cell .code}
``` python
print('Final Result ResNet-18 without Cutout for Test CIFAR-100 Dataset: %.3f' % (final_test_acc_resnet18_cifar100))
print('Final Result ResNet-18 using Cutout for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar100_cutout))
print('Final Result ResNet-18 using Data Augmentation for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_resnet18_cifar100_da))
print('Final Result ResNet-18 using Data Augmentation and  Cutout for CIFAR-100 Test Dataset: %.3f' % (final_test_acc_resnet_cifar100_da_cutout))
```
:::