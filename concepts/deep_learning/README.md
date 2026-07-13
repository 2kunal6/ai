## Models
- U-net:
  - A special type of CNN that is explicitly designed for image segmentation
  - A CNN predicts what is in an image, whereas a U-net predicts what is in an image and where is it located on a pixel-by-pixel basis
- Diffusion Models:
  - Generative AI model that generates data like images by starting from a random noise and gradually removing noise until the image appears
  - This is how stable diffusion creates images from text prompts
  - Steps:
    - Forward Diffusion: take a clear image of a cat for example, and keep adding noise to it until it becomes pure noise 
    - Reverse Diffusion: The model learns to reverse the noising until the original value is recovered
  - GANs generate directly but here it images are not generated directly, it generates images by removing noise
  - Training:
    - take an image and add random noise
    - ask the model to predict the noise added
    - loss: MSE 
  - diffusion models learn to predict and remove noise, which it can then use to create images from prompts
  - Inpainting: fill missing/masked part of an image; masking can be done based on regions
  - diffusion inpainting: add noise in the masked region only; denoise; replace with prompt image
    - negative_prompt: what not to do; example - blurry, red..
- BLIP:
  - image to text model 
  - multimodal
  - trained on various text and image combinations to understand the relationship between the two
  - it can caption images it's never seen before because it uses embeddings of images and corresponding texts; it learns from the proximity of these embeddings
  - for inference we only need to use the image encoder and text decoder parts of the model
- Recurrent Neural Networks


## Miscellaneous
- Model Zoo: collection of pretrained libraries
  - ex: torchvision, huggingface
  - hf's transformers library holds a collection of transformer models
- multimodal model: handle different types of data modalities like text, image etc.
- Neural Networks work best when data is roughly between 0 to 1 or -1 to 1 floating point numbers.
  - Therefore when we load images we cast them to floats and normalize them, for example by dividing by 255 which is the max possible pixel value 
  - another technique is to standardize the data across all RGB channels (0 mean and 1 standard deviation)
  - we also need to do other geometric transformations like cropping, rotating, scaling etc.