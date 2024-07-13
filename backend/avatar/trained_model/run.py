from typing import List, Optional

import torch
from diffusers import DDIMScheduler
from tqdm import tqdm

from backend.avatar.trained_model.model import ConditionedUnet
from backend.utils import logger

SCHEDULER_ID = "google/ddpm-celebahq-256"
WEIGHTS = "./weights/avatar/model.pth"
IMAGE_SIZE = 128
NUM_STEPS = 40

scheduler = DDIMScheduler.from_pretrained(SCHEDULER_ID)
scheduler.set_timesteps(num_inference_steps=NUM_STEPS)
model = ConditionedUnet()
model.load_state_dict(torch.load(WEIGHTS))
model.eval()

props_dict = {
    "Изогнутые брови": {"pos": 0, "value": 1},    "Привлекательность": {"pos": 1, "value": 1},
    "Мешки под глазами": {"pos": 2, "value": 1},  "Лысый": {"pos": 3, "value": 1},
    "Челка": {"pos": 4, "value": 1},              "Большие губы": {"pos": 5, "value": 1},
    "Большой нос": {"pos": 6, "value": 1},        "Черные волосы": {"pos": 7, "value": 1},
    "Светлые волосы": {"pos": 8, "value": 1},     "Размытые": {"pos": 9, "value": 1},
    "Коричневые волосы": {"pos": 10, "value": 1}, "Густые брови": {"pos": 11, "value": 1},
    "Пухлость": {"pos": 12, "value": 1},          "Двойной подбородок": {"pos": 13, "value": 1},
    "Очки": {"pos": 14, "value": 1},              "Эспаньолка": {"pos": 15, "value": 1},
    "Седые волосы": {"pos": 16, "value": 1},      "Сильный макияж": {"pos": 17, "value": 1},
    "Высокие скулы": {"pos": 18, "value": 1},     "Мужчина": {"pos": 19, "value": 1},
    "Рот слегка открыт": {"pos": 20, "value": 1}, "Усы": {"pos": 21, "value": 1},
    "Узкие глаза": {"pos": 22, "value": 1},       "Без бороды": {"pos": 23, "value": 1},
    "Овальное лицо": {"pos": 24, "value": 1},     "Бледная кожа": {"pos": 25, "value": 1},
    "Острый нос": {"pos": 26, "value": 1},        "Залысины": {"pos": 27, "value": 1},
    "Розовые щеки": {"pos": 28, "value": 1},      "Бакенбарды": {"pos": 29, "value": 1},
    "Улыбчивость": {"pos": 30, "value": 1},       "Прямые волосы": {"pos": 31, "value": 1},
    "Волнистые волосы": {"pos": 32, "value": 1},  "В серьгах": {"pos": 33, "value": 1},
    "В шляпе": {"pos": 34, "value": 1},           "Губная помада": {"pos": 35, "value": 1},
    "Ожерелье": {"pos": 36, "value": 1},          "В галстуке": {"pos": 37, "value": 1},
    "Молодость": {"pos": 38, "value": 1},         "Прямые брови": {"pos": 0, "value": -1},
    "Узкие губы": {"pos": 5, "value": -1},        "Маленький нос": {"pos": 6, "value": -1},
    "Четкость": {"pos": 9, "value": -1},          "Редкие брови": {"pos": 11, "value": -1},
    "Стройность": {"pos": 12, "value": -1},       "Без макияжа": {"pos": 17, "value": -1},
    "Женщина": {"pos": 19, "value": -1},          "Широкие глаза": {"pos": 22, "value": -1},
    "Темная кожа": {"pos": 25, "value": -1},      "Грусть": {"pos": 30, "value": -1},
    "Старость": {"pos": 38, "value": -1},         "С бородой": {"pos": 23, "value": -1},
}

woman_vector = torch.FloatTensor([[
     1,  1, -1, -1, -1,  1,  0, -1,  1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,  1, -1,
    -1, -1, -1,  1, -1,  0,  1, -1, -1, -1,
     1,  0,  0, -1, -1,  1, -1, -1,  1
]])

man_vector = torch.FloatTensor([[
    -1,  1, -1, -1, -1, -1,  0,  1, -1, -1,
    -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,
    -1,  1, -1,  1, -1, -1, -1, -1, -1,  1,
     0, -1,  1, -1, -1, -1, -1,  1,  0
]])


def run(prompts: Optional[List[str]] = None):

    if prompts is None:
        vector = man_vector
    else:
        if ("Мужчина" in prompts) or ("С бородой" in prompts) or \
           ("Усы" in prompts) or ("Лысый" in prompts) or \
           ("Залысины" in prompts) or ("Эспаньолка" in prompts) or \
           ("Бакенбарды" in prompts):
            vector = man_vector
        else:
            vector = woman_vector

        for prompt in prompts:
            item = props_dict.get(prompt)
            if item:
                vector[0, item["pos"]] = item["value"]

    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        model_input = scheduler.scale_model_input(x, t)
        with torch.no_grad():
            noise_pred = model(model_input, t, vector)

        scheduler_output = scheduler.step(noise_pred, t, x)
        x = scheduler_output.prev_sample
    x = (x[0].permute(1, 2, 0).clip(-1, 1) * 0.5 + 0.5).numpy()
    logger.warn(x.shape)

    return x
