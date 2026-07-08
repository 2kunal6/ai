## Intro
- python library to build Deep Learning projects
- Tensor:
  - core data structure of pytorch which are multidimensional arrays, similar to numpy arrays
- it has high-performance c++ runtime for faster inference in production
- Capabilities:
  - fast because most of the code is built in c++ and cuda
  - easy to run on CPU and GPU
  - ability of tensors to remember what numerical ops were made on them, so that we can figure out how much an output will change if we change the input a bit.
    - autograd uses this feature

## Implementation
- torch.nn: has core pytorch modules like connected layers, convolution layers, activation functions, loss functions etc.
- Dataset class: to bridge the gap between raw data into tensors so that pytorch can handle it
- DataLoader: to load data in parallel
- torch.optim: for optimizer classes
- torch.distributed: to run across multiple machines and GPU
- torch.compile: optimizes model performance
- executorch: mobile deployment

## Miscellaneous
- Vision Transformers are performing good these days for image classification
  - self attention mechanism capture intricate details in the images
  - we need to transform the image to make it similar to the training data. steps:
    - resize: resizes image keeping the aspect ratio same
    - CenterCrop: crop the center piece to remove borders if any
    - toTensor: convert to tensor
    - Normalize: around some mean and standard deviation
- modules or layers are the building blocks of the deep learning models
  - they can be nested i.e. modules can have submodules
  - typical models are sequential ending with a fully-connected layer of output classes
- squeeze and unsqueeze:
  - squeeze removes dimension of value 1
  - unsqueeze adds deimension of value 1
    - for example if a nn expects input of dimension = 4 (batchsize, color_channels, height, width) and we just have to input it 1 image then we can unsqueeze at dimension=0 for batchsize
- Named Tensors
- transpose:
  - used for image view conversion 
  - need to apply contiguous after the transpose call