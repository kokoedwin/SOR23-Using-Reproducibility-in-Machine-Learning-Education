

::: {.cell .markdown }

## 2. Cutout as a regularization technique

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
### Implementation of Cutout
:::


::: {.cell .code}
```python
# Import necessary libraries
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ColorJitter
```
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

To see how it works, in the following cell, you will upload an image of your choice to this workspace. To prevent any distortion due to resizing, it is advised to use an image that is approximately square in shape, as we will be resizing the image to a square format (100x100 pixels) later on:

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
img = Image.open('./sample.png')

# Resize the image to 100x100
img = img.resize((100, 100))
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
cutout_obj = Cutout(n_holes=1, length=50)

# Apply Cutout to the image
img_tensor_Cutout = cutout_obj(img_tensor)

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

# Define standard data augmentation techniques
transforms_data_augmentation = transforms.Compose([
    RandomHorizontalFlip(),
    RandomCrop(size=(100, 100), padding=4),  # assuming input image is size 100x100
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

# Apply transformations to the image
augmented_img = transforms_data_augmentation(img)

# Display the original and augmented image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(augmented_img)
ax[1].set_title('Augmented Image')
plt.show()

```
:::

