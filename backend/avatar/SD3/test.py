"""
Download weights
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
"""
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "/backend/weights/SD3",
    torch_dtype=torch.float16,
    variant="fp16",
    # device_map="balanced", - for cuda
)

text = "two hugging cats"

image = pipe(
    prompt=text,
    height=512,
    width=512,
    num_inference_steps=10,
    guidance_scale=7.0,
    negative_prompt=None,
    num_images_per_prompt=1,
    output_type="np",  # "pil"
    return_dict=False,  # True
    callback_on_step_end=None,
)

print(image.shape)
