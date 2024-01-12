import inspect
from typing import Callable, List, Optional, Union
from einops import rearrange
import logging
import torch
from PIL import Image
import decord
import glob
import os, cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
import torch.nn.functional as F
from diffusers.utils import is_accelerate_available
from transformers import CLIPImageProcessor, AutoProcessor, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
# from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate
from diffusers import DDPMScheduler
from decord import VideoReader

decord.bridge.set_bridge('torch')
# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image, img_transform):
    image = img_transform(image)
    return image


def init_attention_func(unet):
    #ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276
    def new_attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1,
                                                             -2)) * self.scale
        attn_slice = attention_scores.softmax(dim=-1)
        # compute attention output
        if self.use_last_attn_slice and 'transformer_blocks2' not in self.module_name:
            file_path = os.path.join(
                '/home/gonglitong.glt/workspace/data/video-generation-attn',
                f'{self.noise_level_t}_{self.module_name}.pt')
            attn_slice = torch.load(file_path).float()
            self.use_last_attn_slice = False

        if self.save_last_attn_slice:
            file_path = os.path.join(
                '/home/gonglitong.glt/workspace/data/video-generation-attn',
                f'{self.noise_level_t}_{self.module_name}.pt')
            torch.save(attn_slice.half(), file_path)
            self.save_last_attn_slice = False

        hidden_states = torch.matmul(attn_slice, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def new_sliced_attention(self, query, key, value, sequence_length, dim):

        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads),
            device=query.device,
            dtype=query.dtype)
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[
            0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx],
                             key[start_idx:end_idx].transpose(1, 2)) *
                self.scale)  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)

            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    new_attn_slice = torch.index_select(
                        self.last_attn_slice, -1, self.last_attn_slice_indices)
                    attn_slice = attn_slice * (
                        1 - self.last_attn_slice_mask
                    ) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice

                self.use_last_attn_slice = False

            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False

            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
            module.noise_level_t = None
            module._sliced_attention = new_sliced_attention.__get__(
                module, type(module))
            module._attention = new_attention.__get__(module, type(module))


def use_last_tokens_attention(unet, t, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.noise_level_t = t.item()
            module.use_last_attn_slice = use
            module.module_name = name


def use_last_tokens_attention_weights(unet, t, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use


def use_last_self_attention(unet, t, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.noise_level_t = t.item()
            module.use_last_attn_slice = use
            module.module_name = name


def save_last_tokens_attention(unet, t, save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.noise_level_t = t.item()
            module.save_last_attn_slice = save
            module.module_name = name


def save_last_self_attention(unet, t, save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.noise_level_t = t.item()
            module.save_last_attn_slice = save
            module.module_name = name


def inpaint_image_tensor(masked_image, mask, img_transform, weight_dtype,
                         device):
    """
    masked_image: (f, c, h, w), [-1, 1]
    mask: (1, h, w)
    """

    new_image = (masked_image.permute(0, 2, 3, 1).cpu().numpy() / 2 +
                 0.5) * 255.
    new_image = new_image.astype(np.uint8)
    mask = mask.cpu().numpy().astype(np.uint8)
    res = []
    for frame in new_image:
        # print("frame shape is ", frame.shape)
        # print("mask shape is mask ", mask.shape)
        frame_inpaint = cv2.inpaint(frame, mask[0][0], 5, cv2.INPAINT_TELEA)
        tensor = torch.from_numpy(frame_inpaint).float() / 255.
        res.append(tensor)
    res = torch.stack(res, dim=0)
    res = res.permute(0, 3, 1, 2)
    res = img_transform(res)
    res = res.to(weight_dtype).to(device)
    return res


class StableDiffusionPipelineVideo2VideoMaskC(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(self, vae: AutoencoderKL, unet: UNet2DConditionModel,
                 scheduler: Union[DDIMScheduler, PNDMScheduler,
                                  LMSDiscreteScheduler],
                 scheduler_pre: DDPMScheduler):
        super().__init__()

        if hasattr(scheduler.config,
                   "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file")
            deprecate("steps_offset!=1",
                      "1.0.0",
                      deprecation_message,
                      standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(vae=vae,
                              unet=unet,
                              scheduler=scheduler,
                              scheduler_pre=scheduler_pre)

    def enable_attention_slicing(self,
                                 slice_size: Optional[Union[str,
                                                            int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def cuda_with_minimal_gpu_usage(self):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "Please install accelerate via `pip install accelerate`")

        device = torch.device("cuda")
        self.enable_attention_slicing(1)

        for cpu_offloaded_model in [
                self.unet, self.text_encoder, self.vae, self.safety_checker
        ]:
            cpu_offload(cpu_offloaded_model, device)

    @torch.no_grad()
    def __call__(
        self,
        init_image,
        cond: bool = False,
        strength: float = 0.8,
        batch_size: int = 1,
        num_frames: int = 16,
        height: int = 512,
        width: int = 512,
        fps: int = 25,
        mask_ratio: float = 0.5,
        guidance_scale: float = 7.5,
        copy_raw_images: bool = False,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        latents_dtype=torch.float16,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor],
                                    None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )
        # if cond and strength != 1.0:
        #     raise ValueError("For conditional generation, the value of strength must equal 1.0")
        if (callback_steps is None) or (callback_steps is not None and
                                        (not isinstance(callback_steps, int)
                                         or callback_steps <= 0)):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}.")
        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (batch_size, 4, num_frames, height // 8, width // 8)
        b, c, f, h, w = latents_shape
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        img_transform = T.Compose([
            T.Resize((height, height)),
            T.CenterCrop(height),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        videos = init_image
        init_images = []
        for path in videos[:b]:
            vr = VideoReader(open(path, 'rb'))
            k = vr.get_avg_fps() / fps * min(len(vr), num_frames)
            k = min(len(vr) - 1, k)
            selcted_frame_indexes = np.linspace(0, k, num_frames)
            frames = vr.get_batch(selcted_frame_indexes).permute(0, 3, 1,
                                                                 2).float()
            frames = preprocess(frames / 255., img_transform)
            init_images.append(frames)
        init_images = torch.stack(init_images)

        init_images = init_images.to(device=self.device, dtype=latents_dtype)
        init_images = rearrange(init_images, "b f c h w -> (b f) c h w")
        ori_latents = self.vae.encode(init_images).latent_dist.sample()
        ori_latents = 0.18215 * ori_latents
        ori_latents = rearrange(ori_latents,
                                "(b f) c h w -> b c f h w",
                                b=batch_size,
                                f=num_frames)
        mask_index = torch.linspace(0,
                                    f - 1,
                                    round(f * mask_ratio),
                                    dtype=torch.long,
                                    device=self.device)  # ori mask
        zero_latents = self.vae.encode(torch.zeros_like(init_images[[
            0,
        ]])).latent_dist.sample() * 0.18215
        # zero_input = torch.zeros_like(init_images[[0,]])
        # zero_input[:, 0] = 1.0
        # zero_input[:, 1:] = -1.0
        # zero_latents = self.vae.encode(zero_input).latent_dist.sample() * 0.18215
        latents_cond = zero_latents.view((1, c, 1, h, w)).repeat(
            (b, 1, f, 1, 1))
        uncond_latents_cond = latents_cond.clone()
        # latents_cond = torch.zeros_like(ori_latents)
        latents_cond[:, :, mask_index, :, :] = ori_latents[:, :,
                                                           mask_index, :, :]
        binary_channel = torch.zeros((b, 1, f, h, w),
                                     dtype=latents_dtype,
                                     device=self.device)
        uncond_binary_channel = binary_channel.clone()
        binary_channel[:, :, mask_index, :, :] = 1.0
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn((b, c, f, h, w),
                            generator=generator,
                            device=self.device,
                            dtype=latents_dtype)
        # noise = torch.randn((b, c, 1, h, w), generator=generator, device=self.device, dtype=latents_dtype)
        # noise = noise.repeat((1, 1, f, 1, 1))
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        latents = noise
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            latents_cond = torch.cat([uncond_latents_cond, latents_cond])
            binary_channel = torch.cat([uncond_binary_channel, binary_channel])
        logging.info("begin eval ... ")
        for i, t in enumerate(timesteps):
            if i < 20:
                guidance_scale = 3.0
            else:
                guidance_scale = 1.5 if guidance_scale == 3.0 else 1.5
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = torch.cat(
                [latent_model_input, latents_cond, binary_channel], dim=1)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # latent_model_input = latent_model_input.float()
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, fps=fps).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            if i % 100 == 0:
                logging.info(f"begin eval ... [{i}/{len(timesteps)}]")
        if copy_raw_images:
            latents[:, :, mask_index, :, :] = ori_latents[:, :,
                                                          mask_index, :, :]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        images = self.vae.decode(latents).sample
        images = rearrange(images,
                           "(b f) c h w -> b c f h w",
                           b=batch_size,
                           f=num_frames)

        images = (images / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        videos = []
        for i in range(len(images)):
            image = images[i].cpu().permute(1, 2, 3, 0).numpy()
            if output_type == "pil":
                image = self.numpy_to_pil(image)
            videos.append(image)

        if not return_dict:
            return (videos, )
        logging.info(f"finish eval!")
        return [ImagePipelineOutput(images=x) for x in videos]

    @torch.no_grad()
    def outpainting_with_random_masked_latent_inference_bidirection(
            self,
            ori_image,
            mask_image,
            mask,
            noise_level,
            generator,
            fps,
            use_add_noise=True,
            strength: float = 1.0,
            batch_size: int = 1,
            num_frames: int = 16,
            num_global_frames: int = 16,
            height: int = 256,
            width: int = 256,
            guidance_scale: float = 2.0,
            previous_guidance_scale: float = 1.0,
            num_inference_steps: int = 50,
            eta: float = 0.0,
            latents: Optional[torch.FloatTensor] = None,
            latents_dtype=torch.float16,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor],
                                        None]] = None,
            callback_steps: Optional[int] = 1,
            # mode = 'dense',
            already_outpainted_latents=None,
            copy_already_frame=False,
            replace_gt=False,
            global_frames=None,
            **kwargs):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        if (callback_steps is None) or (callback_steps is not None and
                                        (not isinstance(callback_steps, int)
                                         or callback_steps <= 0)):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}.")

        latents_shape = (batch_size, 4, num_frames, height // 8, width // 8
                         )  # why 4
        b, c, f, h, w = latents_shape

        # bcfhw: 1 4 16 32 32
        # set timesteps

        self.scheduler.set_timesteps(num_inference_steps)
        init_images = ori_image
        init_images = rearrange(init_images,
                                "(b c) f h w -> (b f) c h w",
                                b=batch_size)
        mask_image = rearrange(mask_image,
                               "(b c) f h w -> (b f) c h w",
                               b=batch_size)
        mask = F.interpolate(mask, size=(h, w)).view((b, 1, 1, h, w)).repeat(
            (1, 1, f, 1, 1))
        # print('hha', init_images.size())
        res_latent = self.vae.encode(
            torch.cat([
                init_images, mask_image,
                torch.ones_like(mask_image[0:1]) * -1.0
            ],
                      dim=0)).latent_dist.sample() * 0.18215
        ori_latents = res_latent[:init_images.shape[0]]
        masked_latents = res_latent[init_images.shape[0]:init_images.shape[0] +
                                    mask_image.shape[0]]

        uncond_latents = res_latent[-1:, ...]
        uncond_latents = uncond_latents.view((b, c, 1, h, w)).repeat(
            (1, 1, f, 1, 1))
        uncond_mask = torch.ones_like(mask)

        ori_latents = rearrange(ori_latents,
                                "(b f) c h w -> b c f h w",
                                b=batch_size,
                                f=num_frames)
        masked_latents = rearrange(masked_latents,
                                   "(b f) c h w -> b c f h w",
                                   b=b,
                                   f=f)

        frame_already_outpainted_num = 0
        frame_already_outpainted_list = []
        for idx, already_outpainted_latent in enumerate(
                already_outpainted_latents):
            if already_outpainted_latent != None:
                # print('here')
                masked_latents[:, :, idx] = already_outpainted_latent.to(
                    masked_latents.device)
                mask[:, :, idx] = torch.zeros((h, w)).to(mask.device)

                uncond_mask[:, :, idx] = torch.zeros((h, w)).to(mask.device)
                uncond_latents[:, :, idx] = already_outpainted_latent.to(
                    masked_latents.device)

                frame_already_outpainted_num += 1
                frame_already_outpainted_list.append('*')
            else:
                frame_already_outpainted_list.append('|')
        print(' '.join(frame_already_outpainted_list))

        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, device=self.device)

        noise = torch.randn((b, c, f, h, w),
                            generator=generator,
                            device=self.device,
                            dtype=latents_dtype)

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        # latents = noise
        if use_add_noise:
            latents = self.scheduler.add_noise(ori_latents, noise, timesteps)

        else:
            latents = noise

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        do_classifier_free_guidance = guidance_scale > 1.0
        do_previous_guidance = previous_guidance_scale > 1.0

        if global_frames != None:

            global_frames = self.vae.encode(
                global_frames).latent_dist.sample() * 0.18215
        else:
            global_frames = torch.zeros((16, 4, 32, 32), device=latents.device)

        if do_classifier_free_guidance:
            latents_cond = torch.cat([uncond_latents, masked_latents])
            mask_channel = torch.cat([uncond_mask, mask])
            # previous_latents = torch.cat([previous_latents] * 2)
        else:
            latents_cond = masked_latents
            mask_channel = mask
        if do_previous_guidance:
            latents_cond = torch.cat(
                [latents_cond] * 2
            )  # [uncond_mask_cond_previous, cond_mask_cond_previous, uncond_mask_uncond_previous, cond_mask_uncond_previous]
            mask_channel = torch.cat([mask_channel] * 2)

        # logging.info("begin eval ... ")
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = torch.cat(
                [latent_model_input] *
                2) if do_previous_guidance else latent_model_input
            latent_model_input = torch.cat(
                [latent_model_input, latents_cond, mask_channel], dim=1)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            if do_previous_guidance:
                noise_pred = self.unet(latent_model_input,
                                       t,
                                       fps=fps,
                                       encoder_hidden_states=global_frames,
                                       b=b,
                                       f=num_frames,
                                       g=num_global_frames,
                                       is_empty_img=False,
                                       do_classfier=4).sample
            else:
                noise_pred = self.unet(latent_model_input,
                                       t,
                                       fps=fps,
                                       encoder_hidden_states=global_frames,
                                       b=b,
                                       f=num_frames,
                                       g=num_global_frames,
                                       is_empty_img=False,
                                       do_classfier=2).sample

            if do_classifier_free_guidance and not do_previous_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            elif do_classifier_free_guidance and do_previous_guidance:
                noise_pred_uncond_local_uncond_global, noise_pred_cond_local_uncond_global, noise_pred_uncond_local_cond_global, noise_pred_cond_local_cond_global = noise_pred.chunk(
                    4)
                noise_pred = noise_pred_uncond_local_uncond_global + guidance_scale * (noise_pred_cond_local_uncond_global - noise_pred_uncond_local_uncond_global) + \
                    previous_guidance_scale * (noise_pred_cond_local_cond_global - noise_pred_cond_local_uncond_global)

            elif not do_classifier_free_guidance and not do_previous_guidance:
                pass
            else:
                raise Exception("not implement!")
            # compute the previous noisy sample x_t -> x_t-1

            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            # if i % 100 == 0:
            #     logging.info(f"begin eval ... [{i}/{len(timesteps)}]")

        decoded_latents = latents[:, :, :, :, :]  # [b, c, h, w]

        if copy_already_frame:
            for idx, already_outpainted_latent in enumerate(
                    already_outpainted_latents):
                if already_outpainted_latent != None:
                    # print('here')
                    decoded_latents[:, :, idx] = already_outpainted_latent.to(
                        self.device)
        if replace_gt:
            decoded_latents = masked_latents * (1 -
                                                mask) + decoded_latents * mask

        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # latents = latents.half()
        images = self.vae.decode(latents).sample
        images = rearrange(images,
                           "(b f) c h w -> b c f h w",
                           b=batch_size,
                           f=num_frames)
        images = (images / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        videos = []
        input_images = []

        # from PIL import ImageFilter
        for i in range(len(images)):
            image = images[i].cpu().permute(1, 2, 3, 0).numpy()
            if output_type == "pil":
                image = self.numpy_to_pil(image)
            videos.append(image)

        return [ImagePipelineOutput(images=x) for x in videos], decoded_latents
