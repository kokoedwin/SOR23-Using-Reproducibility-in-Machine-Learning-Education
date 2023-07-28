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
import cv2


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



::: {.cell .markdown}
# Cutout data augmentation

In this notebook, we will reproduce the results of the paper

> DeVries, T. and Taylor, G.W., 2017. Improved regularization of convolutional neural networks with Cutout. arXiv preprint [arXiv:1708.04552](https://arxiv.org/abs/1708.04552). 

We will use the author's implementation of their technique, from [https://github.com/uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout), which is licensed under an Educational Community License version 2.0.

:::



::: {.cell .markdown}

## 1. Learning outcomes

After working through this notebook, you should be able to:

* Describe how Cutout works as a regularization technique,
* Enumerate specific claims (both quantitative claims, qualitative claims, and claims about the underlying mechanism behind a result) from the Cutout paper,
* Execute experiments (following the given procedure) to try and validate each claim about Cutout data augmentation,
* Evaluate whether your own result matches quantitative claims in the Cutout paper (i.e. whether it is within the confidence intervals for each reported numeric result),
* Evaluate whether your own result validates qualitative claims in the Cutout paper,
* Evaluate whether your own results support the author's claim about the underlying mechanism behind the result.

:::



::: {.cell .markdown}
In the sections that follow, we will identify and evaluate claims from the original Cutout paper:

1. Cutout improves the robustness and overall performance of convolutional neural networks.
2. Cutout can be used in conjunction with existing forms of data augmentation and other regularizers to further improve model performance.
3. Cutout aimed to remove maximally activated features in order to encourage the network to consider less prominent features

:::