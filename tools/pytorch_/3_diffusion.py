from diffusers import StableDiffusionInpaintPipeline
import torch
import PIL.Image as Image


device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained("sd2-community/stable-diffusion-2-inpainting")
pipe = pipe.to(device)

img = Image.open("resources/horse.jpg")
mask_img = Image.open("resources/horse_mask.jpg")
prompt = ("A zebra replacing the original horse, same pose, same lighting, background, ultra-realistic photography, high resolution")
negative = "distorted background, blurry, text, watermark"
#1
out = pipe(
    prompt=prompt,
    image=img,
    mask_image=mask_img,
    negative_prompt=negative,
    guidance_scale=7.5,
    strength=0.8,
    generator=torch.Generator(device).manual_seed(42)
)

output = out.images[0]
output.show()