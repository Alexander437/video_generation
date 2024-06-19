from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field, model_validator

from backend.logger import logger


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
        title="Sample rate, возможные варианты: 48000, 24000, 8000"
    )

    @model_validator(mode="before")
    def validate_fqn(cls, values: Dict) -> Dict:
        if values["sample_rate"] not in [48000, 24000, 8000]:
            logger.warning(f"Sample rate {values['sample_rate']} not supported. Set 8000!")
            values["sample_rate"] = 8000
        return values
