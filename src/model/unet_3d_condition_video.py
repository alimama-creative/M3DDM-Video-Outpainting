# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import sys

sys.path.insert(0, "./src/")

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
# from diffusers.modeling_utils import ModelMixin
from diffusers import ModelMixin
from diffusers.utils import BaseOutput, logging
from model.embeddings import TimestepEmbedding, Timesteps
from model.unet_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnDownBlock3D,
    CrossAttnUpBlock2D,
    CrossAttnUpBlock3D,
    DownBlock2D,
    DownBlockP3D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock3DCrossAttn,
    UpBlock2D,
    UpBlockP3D,
    get_down_block,
    get_up_block,
)
from model.layer import Pseudo3DConv

from transformers import (
    CLIPTextConfig, )

from transformers.models.resnet.modeling_resnet import ResNetStage
from transformers.models.resnet.configuration_resnet import ResNetConfig
from einops import rearrange


class LightImageEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet_layer = nn.Sequential(
            ResNetStage(config=ResNetConfig(num_channels=4),
                        in_channels=4,
                        out_channels=8,
                        stride=1),
            ResNetStage(config=ResNetConfig(num_channels=4),
                        in_channels=8,
                        out_channels=16,
                        stride=1),
            ResNetStage(config=ResNetConfig(num_channels=4),
                        in_channels=16,
                        out_channels=32,
                        stride=2),
            ResNetStage(config=ResNetConfig(num_channels=4),
                        in_channels=32,
                        out_channels=64,
                        stride=1),
            ResNetStage(config=ResNetConfig(num_channels=4),
                        in_channels=64,
                        out_channels=128,
                        stride=1),
        )
        self.convolution = nn.Conv2d(128, 320, kernel_size=3,
                                     stride=1)  # padding = 3 // 2

    def forward(self, x):
        out = self.resnet_layer(x)
        out = self.convolution(out)
        #         out = self.linear(out)
        #         out = out.reshape((-1, self.num_image_prompt_tokens, self.projection_dim))
        return out


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet3DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    r"""
    UNet3DConditionModel is a conditional 3D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int`, *optional*): The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str] = ("UpBlock3D", "CrossAttnUpBlock3D",
                                      "CrossAttnUpBlock3D",
                                      "CrossAttnUpBlock3D"),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = None,
        attention_head_dim: int = 8,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv3d(in_channels,
                                 block_out_channels[0],
                                 kernel_size=(1, 3, 3),
                                 padding=(0, 1, 1))
        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos,
                                   freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim,
                                                time_embed_dim)

        self.fps_proj = Timesteps(block_out_channels[0],
                                  flip_sin_to_cos,
                                  freq_shift,
                                  max_period=30)
        # self.fps_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        fps_input_dim = block_out_channels[0]
        self.fps_embedding = TimestepEmbedding(fps_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(
                i + 1,
                len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0],
                                          num_groups=norm_num_groups,
                                          eps=norm_eps)
        self.conv_act = nn.SiLU()
        # self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, 3, padding=1)
        self.conv_out = Pseudo3DConv(dim=block_out_channels[0],
                                     dim_out=out_channels,
                                     kernel_size=3,
                                     padding=1)

        # model_id = "CompVis/stable-diffusion-v1-4"
        # config = DesignBoosterConfig.from_pretrained('/data/fanda.ffd/stable-diffusion-v1-4', subfolder = "text_encoder", local_files_only = True)
        self.global_img_encoder = LightImageEncoder()

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.config.attention_head_dim % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )
        if slice_size is not None and slice_size > self.config.attention_head_dim:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )

        for block in self.down_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

        self.mid_block.set_attention_slice(slice_size)

        for block in self.up_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlockP3D,
                               CrossAttnUpBlock3D, UpBlockP3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        b,
        g,
        f,
        is_empty_img=False,
        do_classfier=1,
        encoder_hidden_states: torch.Tensor = None,
        fps: Union[torch.Tensor, float, int] = 1,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info(
                "Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # pre encode global img
        # if is_empty_img == False:
        #     print('here1')
        #     encoder_hidden_states = self.global_img_encoder(encoder_hidden_states)
        #     print('here2')
        #     # print(self.global_img_encoder.device)
        #     encoder_hidden_states = rearrange(encoder_hidden_states, "(b g) h w -> b (g h) w", b = b, g = g)

        # encoder_hidden_states = encoder_hidden_states.expand(-1, f, -1, -1)
        # else:
        #     pass

        if do_classfier == 2:

            if is_empty_img == True:
                negative_prompt_embeds = torch.zeros_like(
                    encoder_hidden_states)
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, negative_prompt_embeds])
            else:
                negative_prompt_embeds = torch.zeros_like(
                    encoder_hidden_states)
                encoder_hidden_states = torch.cat(
                    [negative_prompt_embeds, encoder_hidden_states])

        if do_classfier == 4:
            negative_prompt_embeds = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states = torch.cat([
                negative_prompt_embeds, negative_prompt_embeds,
                encoder_hidden_states, encoder_hidden_states
            ])

        encoder_hidden_states = self.global_img_encoder(
            encoder_hidden_states)  # b * g, 20, 768
        # b * g, 320, 30, 30
        # b * g, 30 * 30, 320
        # if do_classfier == 2:

        encoder_hidden_states = rearrange(encoder_hidden_states,
                                          "(b g) c h w -> b (g h w) c",
                                          b=b * do_classfier,
                                          g=g)
        # else:
        #     encoder_hidden_states = rearrange(encoder_hidden_states, "(b g) h w -> b (g h) w", b = b, g = g)

        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states.expand(-1, f, -1, -1)

        # if do_classfier == 2:

        #     negative_prompt_embeds = torch.zeros_like(encoder_hidden_states)
        #     encoder_hidden_states = torch.cat([negative_prompt_embeds, encoder_hidden_states])

        # elif do_classfier == 4:
        # print('here3')
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps],
                                     dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)  # [bz, t_emb_dim]

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. fps
        fps_steps = fps
        if not torch.is_tensor(fps_steps):
            # TODO: this requires sync between CPU and GPU. So try to pass fps_steps as tensors if you can
            fps_steps = torch.tensor([fps_steps],
                                     dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(fps_steps) and len(fps_steps.shape) == 0:
            fps_steps = fps_steps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        fps_steps = fps_steps.expand(sample.shape[0])

        femb = self.fps_proj(fps_steps)  # [bz, t_emb_dim]

        # fps_steps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        femb = femb.to(dtype=self.dtype)
        femb = self.fps_embedding(femb)

        # 3. pre-process
        sample = self.conv_in(sample)

        # 4. down
        down_block_res_samples = (sample, )
        for downsample_block in self.down_blocks:
            if hasattr(
                    downsample_block,
                    "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    femb=femb,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample,
                                                       temb=emb,
                                                       femb=femb)

            down_block_res_samples += res_samples
            # print('down', res_samples.size())
            # print('down sample', sample.size())
        # print('here4')
        # 5. mid
        sample = self.mid_block(sample,
                                emb,
                                encoder_hidden_states=encoder_hidden_states,
                                femb=femb)
        # print('mid sample', sample.size())
        # 6. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block,
                       "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    femb=femb,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(hidden_states=sample,
                                        temb=emb,
                                        res_hidden_states_tuple=res_samples,
                                        femb=femb,
                                        upsample_size=upsample_size)
        # 7. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample, )

        return UNet3DConditionOutput(sample=sample)


if __name__ == "__main__":
    model = UNet3DConditionModel.from_config(
        'models/SD-UNet3D/config.json').to("cuda")  # 1299 M params
    model_tiny = UNet3DConditionModel.from_config(
        'models/SD-UNet3D/config_tiny.json').to("cuda")  # 208 M params
    import pdb
    pdb.set_trace()
    inputs = torch.randn(2, 4, 1, 32, 32).to("cuda")
    outputs = model(inputs,
                    timestep=torch.tensor([100, 50], device="cuda"),
                    encoder_hidden_states=None,
                    fps=torch.tensor([25, 3],
                                     device="cuda"))  # [1, 4, 1, 32, 32]
    import pdb
    pdb.set_trace()
    pass
