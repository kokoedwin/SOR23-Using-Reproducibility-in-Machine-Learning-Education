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
# If you're using Google Colab, make sure to set up the runtime environment properly.
# Follow these steps to ensure a smooth experience:

- Step 1: Click on the "Runtime" menu at the top of the Colab interface.

- Step 2: Select "Change runtime type" from the dropdown menu.

- Step 3: In the "Runtime type" section, choose the desired hardware accelerator.
    -       In these experiments, we are going to use "GPU" as it speeds up computations.

Step 4: Click "Save" to apply the changes.

Once you've set up the runtime, you're ready to run the code cells in the notebook!
:::

::: {.cell .markdown}
### For everyone using Google Colab to run these experiments
If you are using Google Colab, here's a step-by-step how to connect with your google drive:

1. On the left sidebar of the Colab notebook interface, you will see a folder icon with the Google Drive logo. Click on this folder icon to open the file explorer.

2. If you haven't connected your Google Drive to Colab yet, it will prompt you to do so. Click the "Mount Drive" button to connect your Google Drive to Colab.

3. Once your Google Drive is mounted, you can use the file explorer to navigate to the file you want to open. Click on the folders to explore the contents of your Google Drive.

4. When you find the file you want to open, click the three dots next to the name of the file in the file explorer. From the options that appear, choose "Copy path." This action will copy the full path of the file to your clipboard. Paste the copy path into the 'current_path' below.
:::


::: {.cell .code}
``` python
current_path ="./"
```
:::

::: {.cell .markdown}

To see how it works, in the following cell, you will upload an image of your choice to this workspace. To prevent any distortion due to resizing, it is advised to use an image that is approximately square in shape, as we will be resizing the image to a square format (100x100 pixels) later on:


To see how Cutout works, let's upload an image and apply Cutout to it. Follow these steps to upload an image in this Google Colab notebook:

1. Click on the folder icon in the left sidebar to open the 'Files' tab.
2. Click the 'Upload to session storage' button (the icon looks like a file with an up arrow).
3. Select the image file from your local machine that you want to upload.
4. Wait for the upload to finish. The uploaded file should now appear in the 'Files' tab.
After the image is uploaded, we can use Python code to load it into our notebook and apply the Cutout augmentation

If you are using Chameleon, here are the steps:


1. Click on the upload icon in the left sidebar.
2. Select the image file from your local machine that you want to upload.
3. Wait for the upload to finish. The uploaded file should now appear in the 'Files' tab.
After the image is uploaded, we can use Python code to load it into our notebook and apply the Cutout augmentation to the image.
:::



::: {.cell .markdown}
In the sections that follow, we will identify and evaluate claims from the original Cutout paper:

1. Cutout improves the robustness and overall performance of convolutional neural networks.
2. Cutout can be used in conjunction with existing forms of data augmentation and other regularizers to further improve model performance.
3. Cutout aimed to remove maximally activated features in order to encourage the network to consider less prominent features

:::
