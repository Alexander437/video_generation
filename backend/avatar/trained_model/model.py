import torch
from diffusers import UNet2DModel, DDIMScheduler
from torch import nn


IMAGE_SIZE = 128
VECTOR_SIZE = 39
CLASS_EMB_SIZE = 16


class ConditionedUnet(nn.Module):
    """
    https://huggingface.co/learn/diffusion-course/unit2/3
    """

    def __init__(
            self,
            img_size=IMAGE_SIZE,
            num_classes=VECTOR_SIZE,
            class_emb_size=CLASS_EMB_SIZE
    ):
        super().__init__()

        self.class_emb = nn.Sequential(
            nn.Dropout(0.1),  # 0.9
            nn.Linear(num_classes, class_emb_size),
            nn.SiLU()
        )
        self.add_cond = nn.Conv2d(3 + class_emb_size, 3, kernel_size=1)
        self.model = UNet2DModel(
            act_fn="silu",
            attention_head_dim=None,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            center_input_sample=False,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            downsample_padding=0,
            flip_sin_to_cos=False,
            freq_shift=1,
            in_channels=3,
            layers_per_block=2,
            mid_block_scale_factor=1,
            norm_eps=1e-06,
            norm_num_groups=32,
            out_channels=3,
            sample_size=img_size,
            time_embedding_type="positional",
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),
        )

    def forward(self, x, t, vectors=None):
        bs, ch, w, h = x.shape

        if vectors is not None:
            class_cond = self.class_emb(vectors)
            # (bs x 16)
            class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1) \
                .expand(bs, class_cond.shape[1], w, h)
            # x is shape (bs, 3, 256, 256) and class_cond is now (bs, 16, 256, 256)

            net_input = torch.cat((x, class_cond), 1)  # (bs, 18, 256, 256)
            x = self.add_cond(net_input)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(x, t).sample  # (bs, 3, 256, 256)

    def init_weights(self, unet_state_dict):
        self.model.load_state_dict(unet_state_dict, strict=True)
