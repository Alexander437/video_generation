import torch
import uvicorn
from fastapi import FastAPI

from speech.router import router as speech_router

torch.set_num_threads(4)  # safe optimal value, i.e. 2 CPU cores

app = FastAPI(
    title="Проект по генерации видео"
)

app.include_router(speech_router)

if __name__ == "__main__":
    uvicorn.run(app="main:app", reload=True)
