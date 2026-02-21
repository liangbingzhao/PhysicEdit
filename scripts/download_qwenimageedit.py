import sys as _sys
import os as _os
# Ensure DiffSynth-Studio is on the Python path regardless of working directory.
_DIFFSYNTH_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../DiffSynth-Studio")
)
if _DIFFSYNTH_DIR not in _sys.path:
    _sys.path.insert(0, _DIFFSYNTH_DIR)
del _sys, _os, _DIFFSYNTH_DIR

import argparse
import os

import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

parser = argparse.ArgumentParser(description="Download Qwen-Image-Edit-2509")
parser.add_argument("--local_model_path", type=str, default=".", help="Local model path")
args = parser.parse_args()

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", local_model_path=args.local_model_path),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", local_model_path=args.local_model_path),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", local_model_path=args.local_model_path),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/", local_model_path=args.local_model_path),
)

image_1 = pipe(prompt="一位少女", seed=0, num_inference_steps=40, height=1328, width=1024)
image_1.save("image1.jpg")

