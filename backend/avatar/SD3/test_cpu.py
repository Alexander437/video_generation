"""
First run
```python

ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
# convert weights
ov_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
ov_pipe.save_pretrained("weights/SD_cpu")
```
"""
from optimum.intel import OVStableDiffusionPipeline

model_id = "/media/alex/Elements/My_projects/video_generation/backend/weights/SD_cpu"
ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id)
ov_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)


def generate(prompt,
             negative_prompt=None,
             num_inference_steps=5,
             guidance_scale=7.5):
    img = ov_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pil",
        return_dict=False,
        height=512,
        width=512,
        num_images_per_prompt=1
    )[0]

    return img
