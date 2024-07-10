import os
from time import strftime

from fastapi import APIRouter
from starlette.responses import FileResponse

from backend.avatar.SD_cpu.run import generate

router = APIRouter(
    prefix="/avatar",
    tags=["avatar"],
)


@router.post("/")
def generate_avatar(text: str):
    upload_directory = "./resources/avatar"
    os.makedirs(upload_directory, exist_ok=True)

    path_to_file = f"{upload_directory}/{strftime('%Y_%m_%d_%H.%M.%S')}.jpg"
    img = generate(prompt=text)
    img.save(path_to_file)
    return FileResponse(path_to_file, media_type="image/jpg", filename=path_to_file)