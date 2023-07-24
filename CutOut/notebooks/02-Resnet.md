
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

::: {.cell .markdown} 
Import the dataset of CIFAR-10 
:::

::: {.cell .code}
``` python
train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
```
:::

::: {.cell .markdown} Create Dataset as Dataloader :::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} Define the model :::

::: {.cell .markdown} This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. :::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name = "cifar10_resnet18"

num_classes = 10
resnet18_cifar10 = ResNet18(num_classes=num_classes)


resnet18_cifar10 = resnet18_cifar10.cuda()
learning_rate = 0.1
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(resnet18_cifar10.parameters(), lr=learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown} Training ResNet-18 withuout Cutout :::

::: {.cell .markdown} This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. :::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        resnet18_cifar10.zero_grad()
        pred = resnet18_cifar10(images)

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

    test_acc = test(test_loader, resnet18_cifar10)
    tqdm.write('test_acc: %.3f' % (test_acc))

    #scheduler.step(epoch)  # Use this line for PyTorch <1.4
    scheduler.step()     # Use this line for PyTorch >=1.4

    
torch.save(resnet18_cifar10.state_dict(), 'checkpoints/' + file_name + '.pt')


final_test_acc_without_cutout = (1 - test(test_loader, resnet18_cifar10))*100
print('Result ResNet-18 without Cutout for Test Dataset: %.3f' % (final_test_acc_without_cutout))
``` 
:::

::: {.cell .markdown}

4.2.1.2. Training ResNet-18 in CF10 with Cutout
::: 

::: {.cell .markdown} Import the library ::: 
::: {.cell .code}
``` python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
``` 
:::


::: {.cell .markdown} Check Cuda GPU availability and set seed number ::: 

::: {.cell .code}
``` python
cuda = torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

seed = 1
torch.manual_seed(seed)
```
:::

::: {.cell .markdown} Image Processing for CIFAR-10 :::

::: {.cell .code}
``` python
# Image Preprocessing

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])

train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)

#Add Cutout to the image transformer piepeline
n_holes = 1
length = 16
train_transform.transforms.append(Cutout(n_holes=n_holes, length=length))


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])
``` 
:::

::: {.cell .markdown} Import the dataset of CIFAR-10 :::

::: {.cell .code}
``` python
train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
```
:::

::: {.cell .markdown} Create Dataset as Dataloader :::

::: {.cell .code}
``` python
# Data Loader (Input Pipeline)
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
```
:::

::: {.cell .markdown} Define the model :::

::: {.cell .markdown} This code block sets up the machine learning model, loss function, optimizer, and learning rate scheduler. :::

::: {.cell .code}
``` python
#file_name will be the used for the name of the file of weight of the model and also the result
file_name = "cifar10_resnet18_Cutout"

num_classes = 10
resnet18_cifar10_cutout = ResNet18(num_classes=num_classes)


resnet18_cifar10_cutout = resnet18_cifar10_cutout.cuda()
learning_rate = 0.1
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(resnet18_cifar10_cutout.parameters(), lr=learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
```
:::

::: {.cell .markdown} Training ResNet-18 with Cutout :::

::: {.cell .markdown} This code runs the training loop for the chosen machine learning model over a specified number of epochs. Each epoch involves a forward pass, loss computation, backpropagation, and parameter updates. It also calculates and displays the training accuracy and cross-entropy loss. At the end of each epoch, the model's performance is evaluated on the test set, and the results are logged and saved. :::

::: {.cell .code}
``` python
epochs = 200
for epoch in range(epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        resnet18_cifar10_cutout.zero_grad()
        pred = resnet18_cifar10_cutout(images)

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

    test_acc = test(test_loader,resnet18_cifar10_cutout)
    tqdm.write('test_acc: %.3f' % (test_acc))
    scheduler.step()     
torch.save(resnet18_cifar10_cutout.state_dict(), 'checkpoints/' + file_name + '.pt')


final_test_acc_with_cutout = (1 - test(test_loader,resnet18_cifar10_cutout))*100
print('Result ResNet-18 with Cutout for Test Dataset: %.3f' % (final_test_acc_with_cutout))
```
:::
