import os
import re
import time
from typing import Optional, Union

import torch
import numpy as np
import simpleaudio as sa
from num2words import num2words
from transliterate import translit

from backend.logger import logger
from backend.settings import settings
from backend.speech.schemas import Speaker


class NeuralSpeaker:
    def __init__(self):
        logger.debug('Initializing speech model')
        start = time.time()
        if not os.path.isfile(settings.TXT_TO_SPEECH_WEIGHTS):
            torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                           settings.TXT_TO_SPEECH_WEIGHTS)

        self.model = torch.package.PackageImporter(settings.TXT_TO_SPEECH_WEIGHTS).load_pickle("tts_models", "model")
        self.model.to(settings.DEVICE)
        logger.info(f'Model ready in {time.time() - start:2f} seconds')

    def speak(
            self,
            text: str,
            speaker: Optional[Speaker] = Speaker["kseniya"],
            save_file: Optional[str] = None,
            sample_rate: int = 8000,
            put_accent: bool = True,
            put_yo: bool = True,
    ) -> Union[np.ndarray, None]:
        """
        https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb#scrollTo=pLTZK0O7JPHT

        Лучше расставить ударения знаком '+':
        `В недрах тундры выдры в г+етрах т+ырят в вёдра ядра к+едров.`

        Можно делать паузы и т.д., как показано в google colab ноутбуке:

        ```
        <speak>
        <p>
          Когда я просыпаюсь, <prosody rate="x-slow">я говорю довольно медленно</prosody>.
          Пот+ом я начинаю говорить своим обычным голосом,
          <prosody pitch="x-high"> а могу говорить тоном выше </prosody>,
          или <prosody pitch="x-low">наоборот, ниже</prosody>.
          Пот+ом, если повезет – <prosody rate="fast">я могу говорить и довольно быстро.</prosody>
          А еще я умею делать паузы любой длины, например, две секунды <break time="2000ms"/>.
          <p>
            Также я умею делать паузы между параграфами.
          </p>
          <p>
            <s>И также я умею делать паузы между предложениями</s>
            <s>Вот например как сейчас</s>
          </p>
        </p>
        </speak>
        ```
        """

        text = translit(text, "ru")
        text = re.sub(r'-?[0-9][0-9,._]*', self.__num2words_ru, text)
        logger.debug(f'text after translit and num2words {text}')
        start = time.time()
        try:
            if save_file is not None:
                self.model.save_wav(
                    text=text,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    put_accent=put_accent,
                    put_yo=put_yo
                )
                logger.info(f'Model applied in {time.time() - start: 2f} seconds')
                return None

            else:
                audio = self.model.apply_tts(
                    text=text,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    put_accent=put_accent,
                    put_yo=put_yo
                )
                audio = audio.numpy()
                audio *= 32767 / np.max(np.abs(audio))
                audio = audio.astype(np.int16)
                wave_obj = sa.WaveObject(audio, 1, 2, sample_rate)
                logger.info(f'Model applied in {time.time() - start: 2f} seconds')
                return wave_obj.audio_data

        except ValueError:
            logger.error("Bad input")
            return

    @staticmethod
    def __num2words_ru(match):
        clean_number = match.group().replace(',', '.')
        return num2words(clean_number, lang='ru')


neural_speaker = NeuralSpeaker()
