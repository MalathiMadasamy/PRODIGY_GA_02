!pip install -q diffusers transformers accelerate torch

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

model_id = "runwayml/stable-diffusion-v1-5"

if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

pipe = pipe.to(device)


prompt = input("Enter your image prompt: ")

image = pipe(prompt).images[0]


plt.imshow(image)
plt.axis("off")
plt.show()


image.save("generated_image.png")

print("Image generated and saved as generated_image.png")