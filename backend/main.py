import sys
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI

sys.path.append(str(Path(__file__).parent / "video" / "SadTalker"))
from video.router import router as video_router

torch.set_num_threads(4)  # safe optimal value, i.e. 2 CPU cores

app = FastAPI(
    title="Проект по генерации видео"
)

app.include_router(video_router)

if __name__ == "__main__":
    uvicorn.run(app="main:app", reload=True)
