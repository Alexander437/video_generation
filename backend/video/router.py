import os
import tempfile
from time import strftime
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from starlette.responses import FileResponse

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
        speaker: Speaker = Speaker["aidar"],
        sample_rate: int = 8000,
):
    neural_speaker = NeuralSpeaker()
    video_generator = get_video_gen("SadTalker")
    with tempfile.TemporaryDirectory(
            dir="/media/alex/Elements/My_projects/video_generation/tmp",
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
        # args = get_args()
        # args.source_image = image
        # args.driven_audio = wav_path
        # args.checkpoint_dir = "./weights/SadTalker"
        # args.bfm_folder = "./weights/SadTalker/BFM_Fitting/"
        # main(args)

    return FileResponse(path_to_video, media_type="video/mp4", filename=path_to_video)
