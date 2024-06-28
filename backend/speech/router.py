import io
import wave

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from backend.logger import logger
from backend.speech.neural_speaker import neural_speaker
from backend.speech.schemas import SpeechRecv

router = APIRouter(
    prefix="/speech",
    tags=["speech"],
)


@router.post("/")
async def speak(request: SpeechRecv):
    audio_data = neural_speaker.speak(
        text=request.text,
        speaker=request.speaker,
        sample_rate=request.sample_rate
    )
    logger.debug(f"audio_data: {audio_data}")
    f = io.BytesIO()
    wav_file_in_memory = wave.open(f, 'w')
    wav_file_in_memory.setnchannels(1)  # mono
    wav_file_in_memory.setsampwidth(2)
    wav_file_in_memory.setframerate(request.sample_rate)
    wav_file_in_memory.writeframes(audio_data)
    wav_file_in_memory.close()
    f.seek(0)
    audio_file_response = StreamingResponse(content=f, media_type="audio/wav")
    audio_file_response.headers["Content-Disposition"] = f'attachment; filename = "speech_audio.wav"'
    return audio_file_response
