import os
from time import strftime
from typing import Optional, List

from fastapi import APIRouter
from matplotlib import pyplot as plt
from starlette.responses import FileResponse

from backend.avatar.trained_model.run import run

router = APIRouter(
    prefix="/avatar",
    tags=["avatar"],
)


@router.post("/")
def generate_avatar(text: Optional[List[str]] = None):
    upload_directory = "./results/avatar"
    os.makedirs(upload_directory, exist_ok=True)

    path_to_file = f"{upload_directory}/{strftime('%Y_%m_%d_%H.%M.%S')}.jpg"
    img = run(prompts=text)
    plt.imsave(path_to_file, img)
    return FileResponse(path_to_file, media_type="image/jpeg", filename=path_to_file)
