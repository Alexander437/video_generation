import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "SadTalker"))

from backend.video.SadTalker import SadTalkerGenerator
from backend.video.SadTalker.settings import sad_talker_settings
from backend.video.video_gen import register_video_gen, get_video_gen

register_video_gen("SadTalker", SadTalkerGenerator(sad_talker_settings))
