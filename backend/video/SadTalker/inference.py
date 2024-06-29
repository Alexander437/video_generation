import os
import shutil
from argparse import Namespace
from time import strftime, time
from typing import Union, Optional

import numpy as np

from backend.utils import logger
from backend.video.SadTalker.inference_args import get_args
from backend.video.SadTalker.settings import SadTalkerSettings
from backend.video.SadTalker.src.facerender.animate import AnimateFromCoeff
from backend.video.SadTalker.src.generate_batch import get_data
from backend.video.SadTalker.src.generate_facerender_batch import get_facerender_data
from backend.video.SadTalker.src.test_audio2coeff import Audio2Coeff
from backend.video.SadTalker.src.utils.init_path import init_path
from backend.video.SadTalker.src.utils.preprocess import CropAndExtract


class SadTalkerGenerator:
    def __init__(
            self,
            settings: Union[SadTalkerSettings, Namespace]
    ):
        self.settings = settings
        self.sadtalker_paths = init_path(
            settings.checkpoint_dir,
            os.path.join(os.path.dirname(__file__), 'src/config'),
            settings.size,
            settings.old_version,
            settings.preprocess
        )

        logger.debug("Initializing SadTalkerGenerator")
        t1 = time()
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.settings.device)
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.settings.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.settings.device)
        logger.info(f"SadTalkerGenerator ready in {time() - t1} seconds")

    def __call__(
            self,
            pic: np.ndarray,
            wav_path: str,
            save_dir: str
    ) -> Optional[str]:

        t1 = time()
        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            pic, first_frame_dir, self.settings.preprocess,
            pic_size=self.settings.size,
            source_image_flag=True
        )

        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if self.settings.ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(self.settings.ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ = self.preprocess_model.generate(
                self.settings.ref_eyeblink,
                ref_eyeblink_frame_dir,
                self.settings.preprocess,
                source_image_flag=False
            )
        else:
            ref_eyeblink_coeff_path = None

        if self.settings.ref_pose is not None:
            if self.settings.ref_pose == self.settings.ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(self.settings.ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ = self.preprocess_model.generate(
                    self.settings.ref_pose,
                    ref_pose_frame_dir,
                    self.settings.preprocess,
                    source_image_flag=False
                )
        else:
            ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(
            first_coeff_path, wav_path,
            self.settings.device,
            ref_eyeblink_coeff_path,
            still=self.settings.still
        )
        coeff_path = self.audio_to_coeff.generate(
            batch, save_dir,
            self.settings.pose_style,
            ref_pose_coeff_path
        )

        # 3dface render
        if self.settings.face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(
                self.settings, self.settings.device,
                first_coeff_path, coeff_path, wav_path,
                os.path.join(save_dir, '3dface.mp4')
            )

        # coeff2video
        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path, wav_path,
            self.settings.batch_size, self.settings.input_yaw,
            self.settings.input_pitch, self.settings.input_roll,
            expression_scale=self.settings.expression_scale,
            still_mode=self.settings.still,
            preprocess=self.settings.preprocess,
            size=self.settings.size
        )

        result = self.animate_from_coeff.generate(
            data, save_dir, pic, crop_info,
            enhancer=self.settings.enhancer,
            background_enhancer=self.settings.background_enhancer,
            preprocess=self.settings.preprocess,
            img_size=self.settings.size
        )

        out_file = f"results/{strftime('%Y_%m_%d_%H.%M.%S')}" + '.mp4'
        shutil.move(result, out_file)
        logger.info(f"SadTalkerGenerator ready in {time() - t1} seconds, generated video: {out_file}")
        return out_file


if __name__ == '__main__':
    args = get_args()
    # torch.backends.cudnn.enabled = False
    # pic may be path or np.ndarray
    video_generator = SadTalkerGenerator(args)

    pic = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    path_to_video = video_generator(
        pic=pic,
        wav_path=audio_path,
        save_dir=save_dir
    )

    if not args.verbose:
        shutil.rmtree(save_dir)
