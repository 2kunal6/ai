from torch import nn
import torch.nn.functional as F
import torch
from torchvision import datasets
import os
from torchvision import transforms
import datetime
from torch import optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # self.conv1_dropout = nn.Dropout2d(p=0.4)
        # self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        # self.conv2_dropout = nn.Dropout2d(p=0.4)
        # self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

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
img, _ = cifar2[0]


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        start_time = datetime.datetime.now()
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            end_time = datetime.datetime.now()
            epoch_duration = (end_time - start_time).total_seconds()
            if epoch == 1 or epoch % 10 == 0:
                print(f'{datetime.datetime.now()}, {epoch}, {loss_train/len(train_loader)}, {epoch_duration}')

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)
model = Net().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
training_loop(n_epochs = 100, optimizer = optimizer, model = model, loss_fn = loss_fn, train_loader = train_loader,)


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print(f'{name} {correct/total}')

validate(model, train_loader, val_loader)

torch.save(model.state_dict(), 'birds_vs_airplanes.pt')

loaded_model = Net()
loaded_model.load_state_dict(torch.load('birds_vs_airplanes.pt'))