from torchvision import datasets
import os
from torchvision import transforms
import torch
import torch.nn as nn

data_path = 'resources/data-unversioned/p1ch7/'
os.makedirs(data_path, exist_ok=True)
tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
tensor_cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())

imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
mean = imgs.view(3, -1).mean(dim=1)
std = imgs.view(3, -1).std(dim=1)
transforms.Normalize(mean=mean, std=std)

transformed_cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]))

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in transformed_cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in tensor_cifar10_val if label in [0, 2]]

conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
print(conv)
print(conv.weight.shape)
print(conv.bias.shape)

with torch.no_grad():
    conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0]])
    conv.bias.zero_()

img, _ = cifar2[0]
output = conv(img.unsqueeze(0))
print(img.unsqueeze(0).shape, output.shape)

pool = nn.MaxPool2d(2)
output = pool(img.unsqueeze(0))
print(img.unsqueeze(0).shape, output.shape)

# won't work because we need to flatten the output before passing to nn.Linear
## TODO: use nn.flatten here
model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.Tanh(), nn.MaxPool2d(2),
nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.Tanh(), nn.MaxPool2d(2),
nn.Linear(8 * 8 * 8, 32), nn.Tanh(), nn.Linear(32, 2))