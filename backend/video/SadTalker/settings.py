from typing import Optional, List

from pydantic import Field
from pydantic_settings import BaseSettings


class SadTalkerSettings(BaseSettings):

    ref_eyeblink: Optional[str] = Field(
        title="path to reference video providing eye blinking",
        default=None
    )
    ref_pose: Optional[str] = Field(
        title="path to reference video providing pose",
        default=None
    )
    checkpoint_dir: str = Field(
        title="path to model weights",
        default='./weights/SadTalker'
    )
    result_dir: str = Field(
        title="path to output",
        default='./results'
    )
    pose_style: int = Field(
        title="input pose style from [0, 46)",
        ge=0, lt=46,
        default=0
    )
    batch_size: int = Field(
        title="the batch size of facerender",
        default=2
    )
    size: int = Field(
        title="the image size of the facerender",
        default=256
    )
    expression_scale: float = Field(
        title="the batch size of facerender",
        default=1.0
    )
    input_yaw: Optional[List[int]] = Field(
        title="the input yaw degree of the user",
        default=None
    )
    input_pitch: Optional[List[int]] = Field(
        title="the input pitch degree of the user",
        default=None
    )
    input_roll: Optional[List[int]] = Field(
        title="the input roll degree of the user",
        default=None
    )
    enhancer: Optional[str] = Field(
        title="Face enhancer, [gfpgan, RestoreFormer]",
        default="gfpgan",
        enum=['gfpgan', 'RestoreFormer'],
    )
    background_enhancer: Optional[str] = Field(
        title="background enhancer, [realesrgan]",
        default=None
    )
    device: str = Field(
        title="Use CUDA or CPU",
        default="cuda",
        enum=['cuda', 'cpu'],
    )
    face3dvis: bool = Field(
        title="generate 3d face and 3d landmarks",
        default=False
    )
    still: bool = Field(
        title="can crop back to the original videos for the full body animation",
        default=False
    )
    preprocess: str = Field(
        title="how to preprocess the images",
        default='crop',
        enum=['crop', 'extcrop', 'resize', 'full', 'extfull']
    )
    verbose: bool = Field(
        title="saving the intermediate output or not",
        default=False
    )
    old_version: bool = Field(
        title="use the pth other than safetensor version",
        default=False
    )
    net_recon: str = Field(
        title="useless",
        default='resnet50',
        enum=['resnet18', 'resnet34', 'resnet50']
    )
    init_path: Optional[str] = Field(
        title="Useless",
        default=None
    )
    use_last_fc: bool = Field(
        title="zero initialize the last fc",
        default=False
    )
    bfm_folder: str = Field(
        title="BFM fitting folder",
        default='./weights/SadTalker/BFM_Fitting/'
    )
    bfm_model: str = Field(
        title="bfm model",
        default='BFM_model_front.mat'
    )
    focal: float = Field(
        title="default focal length",
        default=1015.0
    )
    center: float = Field(
        title="default center",
        default=112.0
    )
    camera_d: float = Field(
        title="default camera distance",
        default=10.0
    )
    z_near: float = Field(
        title="default z near",
        default=5.0
    )
    z_far: float = Field(
        title="default z far",
        default=15.0
    )


sad_talker_settings = SadTalkerSettings()
