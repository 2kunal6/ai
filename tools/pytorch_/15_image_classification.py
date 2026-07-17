from torchvision import datasets
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

data_path = 'resources/data-unversioned/p1ch7/'
os.makedirs(data_path, exist_ok=True)
tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
tensor_cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())

print(type(tensor_cifar10))
img, label = tensor_cifar10[99]

plt.imshow(img.permute(1, 2, 0))
plt.show()

imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
print(imgs.shape)

mean = imgs.view(3, -1).mean(dim=1)
std = imgs.view(3, -1).std(dim=1)
transforms.Normalize(mean=mean, std=std)

transformed_cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]))

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in transformed_cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in tensor_cifar10_val if label in [0, 2]]


# 3072 is 3x32x32 which is the size of one image
# 1024, 512 and 128 randomly chosen number of nodes in the hidden layer
# 2 is number of outputs (airplane or bird) -> first entry is probability of airplane and the second one of bird
model = nn.Sequential(nn.Linear(3072, 1024), nn.Tanh(),
nn.Linear(1024, 512), nn.Tanh(),
nn.Linear(512, 128), nn.Tanh(),
nn.Linear(128, 2))

img, _ = cifar2[0]
img_batch = img.view(-1).unsqueeze(0)
out = model(img_batch)
print(out)

loss = nn.CrossEntropyLoss()
print(loss(out, torch.tensor([label])))
_, index = torch.max(out, dim=1)

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
n_epochs = 5
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        img_tensor = imgs.view(batch_size, -1)
        label_tensor = torch.tensor(labels)
        out = model(img_tensor)
        loss = loss_fn(out, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
    print("Accuracy: %f", correct / total)