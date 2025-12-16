"""
Test PanoWan.
"""

from pathlib import Path

import click
import torch

from ..data import save_video
from ..models.model_manager import ModelManager
from ..pipelines.wan_video import WanVideoPipeline

DEFAULT_PROMPT = """Stunning panoramic underwater shot of a vibrant coral reef
ecosystem brimming with marine life. Colorful fish dart effortlessly among
intricate coral formations, soft rays of sunlight filter through the crystal-clear
waters, creating mesmerizing patterns on the ocean floor. Wide-angle capturing
vivid hues and abundant biodiversity.""".replace("\n", " ")

DEFAULT_NEGATIVE_PROMPT = """overly colorful, overexposed, static, blurred details,
subtitles, style, artwork, painting, still image, overall grayish, worst quality,
low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly
drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers,
static scene, cluttered background, three legs, many people in background, walking
backwards, people at bottom of frame, person holding camera""".replace("\n", " ")


def test(
    wan_model_path: str,
    lora_checkpoint_path: str,
    output_path: str,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int = 50,
    seed: int = 0,
    tiled: bool = True,
    width: int = 896,
    height: int = 448,
):
    """
    Test PanoWan pipeline.

    Args:
        wan_model_path: The path to the Wan2.1-T2V-1.3B model.
        lora_checkpoint_path: The path to the lora checkpoint.
        prompt: The prompt to generate the video.
        negative_prompt: The negative prompt to generate the video.
        num_inference_steps: The number of inference steps.
        seed: The seed for the random number generator.
        tiled: Whether to use tiled generation.
        width: The width of the video.
        height: The height of the video.
    """
    assert width == height * 2, "Width must be twice the height"

    # Load models
    wan_model_path = Path(wan_model_path)
    if not wan_model_path.exists():
        raise FileNotFoundError(f"Wan2.1-T2V-1.3B model not found at {wan_model_path}")
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            str(wan_model_path / "diffusion_pytorch_model.safetensors"),
            str(wan_model_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            str(wan_model_path / "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA checkpoint
    lora_checkpoint_path = Path(lora_checkpoint_path)
    if not lora_checkpoint_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found at {lora_checkpoint_path}")
    model_manager.load_lora(str(lora_checkpoint_path), lora_alpha=1.0)

    # Create pipeline
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )

    # Enable VRAM management
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # Generate video
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        tiled=tiled,
        width=width,
        height=height,
    )

    # Save video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(
        video, str(output_path), fps=15, quality=10, ffmpeg_params=["-crf", "18"]
    )
    print(f"Video saved to {output_path}")


@click.command()
@click.option(
    "--wan-model-path",
    type=str,
    default="models/Wan-AI/Wan2.1-T2V-1.3B",
    help="The path to the Wan2.1-T2V-1.3B model.",
)
@click.option(
    "--lora-checkpoint-path",
    type=str,
    default="models/PanoWan/latest-lora.ckpt",
    help="The path to the lora checkpoint.",
)
@click.option(
    "--output-path",
    type=str,
    default="outputs/video.mp4",
    help="The path to save the video.",
)
@click.option(
    "--prompt",
    type=str,
    default=DEFAULT_PROMPT,
    help="The prompt to generate the video.",
)
@click.option(
    "--negative-prompt",
    type=str,
    default=DEFAULT_NEGATIVE_PROMPT,
    help="The negative prompt to generate the video.",
)
@click.option(
    "--num-inference-steps",
    type=int,
    default=50,
    help="The number of inference steps.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="The seed for the random number generator.",
)
@click.option(
    "--tiled",
    type=bool,
    default=True,
    help="Whether to use tiled generation.",
)
@click.option("--width", type=int, default=896, help="The width of the video.")
@click.option("--height", type=int, default=448, help="The height of the video.")
def main(
    wan_model_path: str,
    lora_checkpoint_path: str,
    output_path: str,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps: int = 50,
    seed: int = 0,
    tiled: bool = True,
    width: int = 896,
    height: int = 448,
):
    """
    Entry point for PanoWan pipeline.
    """
    test(
        wan_model_path,
        lora_checkpoint_path,
        output_path,
        prompt,
        negative_prompt,
        num_inference_steps,
        seed,
        tiled,
        width,
        height,
    )


if __name__ == "__main__":
    main()
