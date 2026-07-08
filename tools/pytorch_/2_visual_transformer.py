from torchvision import transforms
from PIL import Image
import torch
from torchvision import models

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img = Image.open("resources/bobby.jpg")
img.show()

img_t = preprocess(img)

batch_t = torch.unsqueeze(img_t, 0)

vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
vit.eval()
out = vit(batch_t)
print(out.shape)
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

with open('resources/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# .item() is to get the value of tensor because those functions return tensor
print(labels[index.item()])
print(percentage[index.item()].item())

_, indices = torch.sort(out, descending=True)
top_5 = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print(top_5)