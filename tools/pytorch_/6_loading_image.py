import imageio.v2 as imageio
import torch

img_arr = imageio.imread('resources/bobby.jpg')
print(img_arr.shape)

img = torch.from_numpy(img_arr)
img = img.permute(2, 0, 1)
print(img.shape)