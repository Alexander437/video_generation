import torch
import uvicorn
from fastapi import FastAPI

from video.router import router as video_router
# from speech.router import router as speech_router

torch.set_num_threads(4)  # safe optimal value, i.e. 2 CPU cores

app = FastAPI(
    title="Проект по генерации видео"
)

app.include_router(video_router)
# app.include_router(speech_router)

if __name__ == "__main__":
    # При reload=True модели будут инициализированы дважды
    # В моем случае это может привести к cuda out of memory
    uvicorn.run(app="main:app", reload=False)
