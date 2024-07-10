import os
import tempfile
from time import strftime
from typing import Annotated

import aiofiles
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from starlette.responses import FileResponse, JSONResponse

from backend.speech.neural_speaker import NeuralSpeaker
from backend.speech.schemas import Speaker
from backend.video import get_video_gen


router = APIRouter(
    prefix="/video",
    tags=["video"],
)


@router.post("/")
async def generate_video(
        img_file: Annotated[UploadFile, File()],
        text: str = "Привет! Как дела?",
        speaker: Speaker = "aidar",
        sample_rate: int = 8000,
):
    neural_speaker = NeuralSpeaker()
    video_generator = get_video_gen("SadTalker")
    with tempfile.TemporaryDirectory(
            dir="./tmp",
    ) as tmpdirname:

        wav_path = os.path.join(tmpdirname, f"{strftime('%Y_%m_%d_%H.%M.%S')}.wav")
        neural_speaker.speak(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate,
            save_file=wav_path,
        )

        img_bytes = await img_file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        path_to_video = video_generator(image, wav_path, tmpdirname)

    return FileResponse(path_to_video, media_type="video/mp4", filename=path_to_video)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    upload_directory = "./tmp/uploads"
    os.makedirs(upload_directory, exist_ok=True)

    file_path = os.path.join(upload_directory, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    return JSONResponse(content={"filename": file.filename}, status_code=200)
