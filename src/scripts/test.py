import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData, load_state_dict
from modelscope import snapshot_download
import os
import numpy as np
from PIL import Image

width = 896
height = 448

# Download models
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
model_manager.load_lora("models/lightning_logs/final/checkpoints/epoch=180-step=15204.ckpt", lora_alpha=1.0)

pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Text-to-video
video = pipe(
    prompt="Stunning panoramic underwater shot of a vibrant coral reef ecosystem brimming with marine life. Colorful fish dart effortlessly among intricate coral formations, soft rays of sunlight filter through the crystal-clear waters, creating mesmerizing patterns on the ocean floor. Wide-angle capturing vivid hues and abundant biodiversity.",
    negative_prompt="overly colorful, overexposed, static, blurred details, subtitles, style, artwork, painting, still image, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static scene, cluttered background, three legs, many people in background, walking backwards, people at bottom of frame, person holding camera",
    num_inference_steps=50,
    seed=0, tiled=True,
    width=width, height=height
)
save_video(video, "underwater_reef.mp4", fps=15, quality=10, ffmpeg_params=['-crf', '18'])
