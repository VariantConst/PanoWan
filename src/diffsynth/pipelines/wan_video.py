import math
import os
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from ..models.model_manager import ModelManager
from ..models.wan_video_dit import RMSNorm, WanModel, sinusoidal_embedding_1d
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_text_encoder import (
    T5LayerNorm,
    T5RelativeEmbedding,
    WanTextEncoder,
)
from ..models.wan_video_vae import CausalConv3d, RMS_norm, Upsample, WanVideoVAE
from ..prompters.wan_prompter import WanPrompter
from ..schedulers.flow_match import FlowMatchScheduler
from ..vram_management import (
    AutoWrappedLinear,
    AutoWrappedModule,
    enable_vram_management,
)
from .base import BasePipeline


class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16

    def _erp_warp(self, latents: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = latents.shape
        latents2d = latents.permute(0, 2, 1, 3, 4).reshape(B * T * C, 1, H, W)
        y = torch.linspace(
            -1, 1, 2 * H + 1, device=latents.device, dtype=latents.dtype
        )[1:-1:2, None]  # y of the center points
        x = torch.linspace(
            -1, 1, 2 * W + 1, device=latents.device, dtype=latents.dtype
        )[None, 1:-1:2]  # x of the center points
        x = x * torch.sin((y * 0.5 + 0.5) * math.pi)
        grid = (
            torch.stack((x.expand(H, W), y.expand(H, W)), dim=-1)
            .unsqueeze(0)
            .repeat(2 * latents2d.shape[0], 1, 1, 1)
        )
        warped, warped_x2 = F.grid_sample(
            torch.cat([latents2d, latents2d**2]),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).chunk(2)
        # In case of I.I.D. x ~ N(0,1), E(x^2) = E(x)^2 + Var(x) = Var(x)
        # Interpolation of Var(x) does not change the variance
        warped = torch.sign(warped) * (warped_x2**0.5)
        # x,y are coords of points' center <=> -1 & 1 = corner points <=> align_corners=False
        warped = warped.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return warped

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width).to(image.device),
            ],
            dim=1,
        )
        y = self.vae.encode(
            [vae_input.to(dtype=self.torch_dtype, device=self.device)],
            device=self.device,
        )[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(
        self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        latents = self.vae.encode(
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return latents

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        frames = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        latent_padding_size=2,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(
                f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
            )

        # Tiler parameters
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        # Initialize noise
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        noise = self._erp_warp(noise)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(
                dtype=self.torch_dtype, device=self.device
            )
            latents = self.encode_video(input_video, **tiler_kwargs).to(
                dtype=self.torch_dtype, device=self.device
            )
            latents = self.scheduler.add_noise(
                latents, noise, timestep=self.scheduler.timesteps[0]
            )
        else:
            latents = noise

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # TeaCache
        tea_cache_posi = {
            "tea_cache": TeaCache(
                num_inference_steps,
                rel_l1_thresh=tea_cache_l1_thresh,
                model_id=tea_cache_model_id,
            )
            if tea_cache_l1_thresh is not None
            else None
        }
        tea_cache_nega = {
            "tea_cache": TeaCache(
                num_inference_steps,
                rel_l1_thresh=tea_cache_l1_thresh,
                model_id=tea_cache_model_id,
            )
            if tea_cache_l1_thresh is not None
            else None
        }

        # Denoise
        self.load_models_to_device(["dit"])
        sum_shift = 0
        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps)
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )

            # Inference
            noise_pred_posi = model_fn_wan_video(
                self.dit,
                latents,
                timestep=timestep,
                **prompt_emb_posi,
                **image_emb,
                **extra_input,
                **tea_cache_posi,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit,
                    latents,
                    timestep=timestep,
                    **prompt_emb_nega,
                    **image_emb,
                    **extra_input,
                    **tea_cache_nega,
                )
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], latents
            )

            # Latent random shift
            random_shift = torch.randint(0, latents.size(-1), (1,)).item()
            sum_shift += random_shift
            latents = torch.roll(latents, shifts=random_shift, dims=-1)

        latents = torch.roll(latents, shifts=-sum_shift % latents.size(-1), dims=-1)

        # Add latent padding
        if latent_padding_size > 0:
            left_pad = latents[..., -latent_padding_size:]
            right_pad = latents[..., :latent_padding_size]
            latents = torch.cat([left_pad, latents, right_pad], dim=-1)

        # Decode
        self.load_models_to_device(["vae"])
        self.vae = self.vae.to(dtype=self.torch_dtype, device=self.device)
        frames = self.decode_video(latents, **tiler_kwargs)

        # Remove latent padding
        if latent_padding_size > 0:
            frame_padding_size = latent_padding_size * 8
            frames = frames[..., frame_padding_size:-frame_padding_size]

        self.load_models_to_device([])

        video = self.tensor2video(frames[0])  # list of np.ndarray (H, W, C)

        return video


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            "Wan2.1-T2V-14B": [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            "Wan2.1-I2V-14B-480P": [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ],
            "Wan2.1-I2V-14B-720P": [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    **kwargs,
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = dit.patchify(x)

    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        # blocks
        for block in dit.blocks:
            x = block(x, context, t_mod, freqs)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
