
::: {.cell .markdown}
##### 4.3.1.3.2. Compare the Qualitative Claims usinng Grad-CAM

###### What is Grad-CAM?
Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that provides visual explanations for decisions made by Convolutional Neural Network (CNN) models. It uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

Grad-CAM is not limited to a specific architecture, it can be applied to a wide range of CNN models without any changes to their existing structure or requiring re-training. It's also class-discriminative, allowing it to effectively manage multi-label scenarios.

By visualizing the model's focus areas with Grad-CAM, we can assess how effectively Cutout is encouraging the model to use a broader range of features. For example, if a model trained with Cutout still primarily focuses on a single region, that might suggest the Cutout squares are too small, or not numerous enough. Conversely, if the focus areas are well spread across the image, it would confirm that Cutout is indeed pushing the model to generalize better.

If you want to understand more about Grad-CAM? Check this paper (https://arxiv.org/abs/1610.02391)
:::

::: {.cell .code}
``` python
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

        # Register hooks for Grad-CAM
        self.gradients = None
        self.activations = None
        self.layer4.register_forward_hook(self._store_activations_hook)
        self.layer4.register_backward_hook(self._store_gradients_hook)

    def _store_activations_hook(self, module, input, output):
        self.activations = output

    def _store_gradients_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

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

```
:::


::: {.cell .code}
``` python

model = ResNet18(num_classes=10)
model.load_state_dict(torch.load("checkpoints/cifar10_resnet18.pt"))
model.eval()

model_co = ResNet18(num_classes=10)
model_co.load_state_dict(torch.load("checkpoints/cifar10_resnet18_Cutout.pt"))
model_co.eval()
```
:::

::: {.cell .markdown}
Let's try to see the result from the testloader of CIFAR-10 dataset
:::

::: {.cell .code}
``` python
import torchvision

transform_dataset = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_dataset)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

```
:::

::: {.cell .code}
``` python
cifar_classes = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
```
:::

::: {.cell .code}
``` python
# Get a batch from the testloader
images, labels = next(iter(testloader))
input_tensor = images  # As your batch_size is 1, you will have a single image here

# Forward pass
model.zero_grad()
output = model(input_tensor)

model_co.zero_grad()
output_co = model_co(input_tensor)

# Get the index of the max log-probability
target = output.argmax(1)
output.max().backward()

target_co = output_co.argmax(1)
output_co.max().backward()

# Map the predicted class indices to the class labels
predicted_class = cifar_classes[target.item()]
predicted_class_co = cifar_classes[target_co.item()]


# Get the gradients and activations
gradients = model.gradients.detach().cpu()
activations = model.activations.detach().cpu()

gradients_co = model_co.gradients.detach().cpu()
activations_co = model_co.activations.detach().cpu()


# Calculate the weights
weights = gradients.mean(dim=(2, 3), keepdim=True)

weights_co = gradients_co.mean(dim=(2, 3), keepdim=True)

# Calculate the weighted sum of activations (Grad-CAM)
cam = (weights * activations).sum(dim=1, keepdim=True)
cam = F.relu(cam)  # apply ReLU to the heatmap
cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)
cam = cam.squeeze().numpy()

cam_co = (weights_co * activations_co).sum(dim=1, keepdim=True)
cam_co = F.relu(cam_co)  # apply ReLU to the heatmap
cam_co = F.interpolate(cam_co, size=(32, 32), mode='bilinear', align_corners=False)
cam_co = cam_co.squeeze().numpy()


# Normalize the heatmap
cam -= cam.min()
cam /= cam.max()

cam_co -= cam_co.min()
cam_co /= cam_co.max()

# Since the images from the dataloader are normalized, you have to denormalize them before plotting
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
img = images.squeeze().detach().cpu() * std[..., None, None] + mean[..., None, None]
img = img.permute(1, 2, 0).numpy()

# Superimpose the heatmap onto the original image
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
superimposed_img = heatmap * 0.4 + img * 255

heatmap_co = cv2.applyColorMap(np.uint8(255 * cam_co), cv2.COLORMAP_JET)
heatmap_co = cv2.cvtColor(heatmap_co, cv2.COLOR_BGR2RGB)
superimposed_img_co = heatmap_co * 0.4 + img * 255

class_label = str(labels.item())

# Display the original image and the Grad-CAM
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(img)
ax[0].set_title('Original Image (Class: ' + cifar_classes[int(class_label)] + ')')
ax[0].axis('off')
ax[1].imshow(superimposed_img / 255)
ax[1].set_title('Grad-CAM: ' + predicted_class)
ax[1].axis('off')
ax[2].imshow(superimposed_img_co / 255)
ax[2].set_title('Grad-CAM with Cutout:'+  predicted_class_co)
ax[2].axis('off')
plt.show()

```
:::

::: {.cell .code}
``` python

```
:::



::: {.cell .markdown}
Now you  can try to load your image, preprocess it and convert it into a PyTorch tensor. 
Choose an image that is in the CIFAR-10 classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks).
The preprocessing steps should be the same as the ones you used for training your model. 
Let's say you have an image `image.jpg`:
:::


::: {.cell .code}
``` python
from PIL import Image
from torchvision import transforms

# Load the image
image_path = "image.jpg"
image = Image.open(image_path)

# Define the transformations: resize, to tensor, normalize (replace the mean and std with values you used for training)
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess the image
input_tensor = preprocess(image)
input_tensor = input_tensor.unsqueeze(0)  # add batch dimension.  C,H,W => B,C,H,W

```
:::

::: {.cell .markdown}
Apply Grad-CAM
:::

::: {.cell .code}
``` python
# Forward pass
model.zero_grad()
output = model(input_tensor)

model_co.zero_grad()
output_co = model_co(input_tensor)

# Get the index of the max log-probability
target = output.argmax(1)
output.max().backward()

target_co  = output_co .argmax(1)
output_co .max().backward()

# Get the gradients and activations
gradients = model.gradients.detach().cpu()
activations = model.activations.detach().cpu()

gradients_co  = model_co.gradients.detach().cpu()
activations_co  = model_co.activations.detach().cpu()

# Calculate the weights
weights = gradients.mean(dim=(2, 3), keepdim=True)

weights_co = gradients_co.mean(dim=(2, 3), keepdim=True)

# Calculate the weighted sum of activations (Grad-CAM)
cam = (weights * activations).sum(dim=1, keepdim=True)
cam = F.relu(cam)  # apply ReLU to the heatmap
cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)
cam = cam.squeeze().numpy()

cam_co = (weights_co * activations_co).sum(dim=1, keepdim=True)
cam_co = F.relu(cam_co)  # apply ReLU to the heatmap
cam_co = F.interpolate(cam_co, size=(32, 32), mode='bilinear', align_corners=False)
cam_co = cam_co.squeeze().numpy()

# Normalize the heatmap
cam -= cam.min()
cam /= cam.max()

cam_co -= cam_co.min()
cam_co /= cam_co.max()

```
:::
::: {.cell .markdown}
Visualize the image and the Grad-CAM heatmap
:::


::: {.cell .code}
``` python
import matplotlib.pyplot as plt
import cv2

# Load the original image
img = cv2.imread(image_path)
img = cv2.resize(img, (32, 32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Superimpose the heatmap onto the original image
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
superimposed_img = heatmap * 0.4 + img

# Superimpose the heatmap onto the original image with cutout
heatmap_co = cv2.applyColorMap(np.uint8(255 * cam_co), cv2.COLORMAP_JET)
heatmap_co = cv2.cvtColor(heatmap_co, cv2.COLOR_BGR2RGB)
superimposed_img_co = heatmap_co * 0.4 + img

# Display the original image and the Grad-CAM
fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis("off")
ax[1].imshow(superimposed_img / 255)
ax[1].set_title('Grad-CAM')
ax[1].axis("off")
ax[2].imshow(superimposed_img_co / 255)
ax[2].set_title('Grad-CAM with Cutout')
ax[2].axis("off")
plt.show()

```
:::