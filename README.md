# Stable Diffusion Text-to-Image

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
run these commands from the repository root:

Windows:
```powershell
cd StableDiffusion
uv sync
uv run python Simple_SD_Text2img.py "A lighthouse beside the sea at sunrise"
```
Linux:
```bash
cd StableDiffusion
uv sync
uv run python Simple_SD_Text2img.py "A lighthouse beside the sea at sunrise"
```

The image is saved to `output.png` in the current working directory.
The first run downloads `CompVis/stable-diffusion-v1-4` from Hugging Face
(several GB); later runs reuse the cache. Review the model's license before
use. If access requires authentication, accept the model's terms on Hugging
Face and run `uvx --from huggingface_hub hf auth login` locally.

```powershell
uv run python Simple_SD_Text2img.py "A watercolor mountain lake" --output images/lake.png --seed 42 --steps 30
uv run python Simple_SD_Text2img.py "A lighthouse beside the sea" --device cpu
uv run python Simple_SD_Text2img.py --help
```

Use `--model` for another compatible Stable Diffusion model ID or a local
Diffusers model directory. `--guidance-scale` defaults to 7.5 and `--steps`
to 50. A seed helps repeat generation in the same environment; results
can differ across hardware and dependency versions. Output must be PNG.

## Environment

This project targets Python 3.14; uv can download it if necessary. The
explicit `pytorch-xpu` index and platform-specific sources follow the parent
project's example. Only image-generation dependencies are included, without
Jupyter, widgets, audio packages, or Intel Extension for PyTorch.

`--device auto` selects Intel XPU when available, otherwise CPU. XPU uses
float16 and needs a supported Intel GPU and current drivers. CPU uses
float32 and is substantially slower. Explicit `--device xpu` fails if XPU
is unavailable. This configuration does not install NVIDIA CUDA builds.
The pipeline's default safety checker remains enabled.
