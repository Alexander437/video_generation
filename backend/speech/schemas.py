from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field, model_validator

from backend.utils import logger


class Speaker(str, Enum):
    aidar = "aidar"
    baya = "baya"
    kseniya = "kseniya"
    xenia = "xenia"
    eugene = "eugene"
    random = "random"


class SpeechRecv(BaseModel):
    text: str = Field(
        min_length=3,
        title="Текст, который будет произноситься"
    )
    speaker: Speaker = Field(
        title="Возможные варианты: aidar, baya, kseniya, xenia, eugene, random"
    )
    sample_rate: int = Field(
        default=8000,
        title="Sample rate, возможные варианты: 48000, 24000, 8000",
        enum=[48000, 24000, 8000],
    )
