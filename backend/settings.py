import os

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    TXT_TO_SPEECH_WEIGHTS: str = os.getenv("TXT_TO_SPEECH_WEIGHTS", "./weights/speech.pt")

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
