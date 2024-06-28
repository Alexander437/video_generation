import wave
import tempfile
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from backend.logger import logger
from backend.speech.neural_speaker import neural_speaker
from backend.speech.schemas import Speaker
from backend.video.SadTalker.inference import main
from backend.video.SadTalker.inference_args import get_args

router = APIRouter(
    prefix="/video",
    tags=["video"],
)


@router.post("/")
async def generate_video(
        img_file: Annotated[UploadFile, File()],
        text: str = "Привет! Как дела?",
        speaker: Speaker = Speaker["aidar"],
        sample_rate: int = 8000,
):
    audio_data = neural_speaker.speak(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate
    )
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
        with wave.open(temp_wav_file, 'w') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        temp_wav_path = temp_wav_file.name

    img_bytes = await img_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    args = get_args()
    args.source_image = image
    args.driven_audio = temp_wav_path
    args.checkpoint_dir = "./weights/SadTalker"
    args.bfm_folder = "./weights/SadTalker/BFM_Fitting/"
    main(args)

    # Удаление временного файла
    # os.remove(temp_wav_path)

    # video_path = f"results/{video_filename}"
    # return FileResponse(video_path, media_type="video/mp4", filename=video_filename)

    return {
        "img_shape": image.shape,
    }
