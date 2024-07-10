import torch
import sys
sys.path.append("backend/avatar/Kandinsky-3/kandinsky3")
from kandinsky3 import get_T2I_Flash_pipeline

device_map = {
    'unet': torch.device('cpu'),
    'text_encoder': torch.device('cpu'),
    'movq': torch.device('cpu')
}
dtype_map = {
    'unet': torch.float16,
    'text_encoder': torch.float16,
    'movq': torch.float16,
}

t2i_pipe = get_T2I_Flash_pipeline(
    device_map, dtype_map, cache_dir='./backend/weights/Kandinsky'
)

res = t2i_pipe(
    "два длинных кота обнимаются",
    negative_text=None,
    guidance_scale=4.0,
    steps=20,
    width=512,
    height=512,
)

print(len(res))
