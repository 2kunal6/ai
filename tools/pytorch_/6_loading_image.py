import imageio.v2 as imageio
import torch

img_arr = imageio.imread('resources/bobby.jpg')
print(img_arr.shape)

img = torch.from_numpy(img_arr)
img = img.permute(2, 0, 1)
print(img.shape)


import os
data_dir = 'resources/cats/images'
png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
batch = torch.zeros(len(png_files), 3, 256, 256, dtype=torch.uint8)

for i, filename in enumerate(png_files):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3] # we need just the first 3 channels
    batch[i] = img_t

batch = batch.float()
batch /= 255.0

n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std