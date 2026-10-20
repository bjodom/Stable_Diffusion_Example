"""Generate a PNG from a text prompt using Stable Diffusion and PyTorch."""

import argparse
import math
from pathlib import Path
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Text describing the image to generate.")
    parser.add_argument("--output", type=Path, default=Path("output.png"))
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4",
                        help="Hugging Face model ID or local Diffusers model directory.")
    parser.add_argument("--device", choices=("auto", "xpu", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, help="Optional seed for repeatable generation.")
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("prompt must not be blank")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error("--guidance-scale must be finite and nonnegative")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must have a .png extension")
    return args


def generate_image(args):
    import torch
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPTextConfig, CLIPTextModel

    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    if args.device == "xpu" and not xpu_available:
        raise RuntimeError("Intel XPU was requested but is not available to PyTorch.")
    device = ("xpu" if xpu_available else "cpu") if args.device == "auto" else args.device
    dtype = torch.float16 if device == "xpu" else torch.float32
    print(f"Using device: {device}")
    if device == "cpu":
        print("CPU inference can be slow.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {args.model}")
    text_config = CLIPTextConfig.from_pretrained(args.model, subfolder="text_encoder")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", config=text_config, dtype=dtype,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, text_encoder=text_encoder, torch_dtype=dtype,
    )
    pipe.to(device)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
    if result.nsfw_content_detected is not None and any(result.nsfw_content_detected):
        raise RuntimeError("The model's safety checker blocked the generated image.")
    result.images[0].save(args.output, format="PNG")
    print(f"Saved image: {args.output.resolve()}")
    return args.output


def main(argv=None):
    args = parse_args(argv)
    try:
        generate_image(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())