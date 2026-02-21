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

from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image_physical import QwenImagePhysicPipeline, ModelConfig

def calculate_dimensions(target_area, ratio):
    import math
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


def resize_image(image, target_area=1024*1024):
    width, height = calculate_dimensions(target_area, image.size[0] / image.size[1])
    return image.resize((width, height))

def load_finetuned_into_pipe(pipe: QwenImagePhysicPipeline, ckpt_path: str):
    if not ckpt_path:
        print("[INFER][ckpt] No checkpoint path provided, skip loading.")
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[INFER][ckpt] checkpoint not found: {ckpt_path}")

    print(f"[INFER][ckpt] Loading finetuned weights from {ckpt_path}")
    full_state = load_state_dict(ckpt_path)

    lora_state = {k: v for k, v in full_state.items()
                  if ("lora_A" in k or "lora_B" in k)}
    non_lora_state = {k: v for k, v in full_state.items()
                      if k not in lora_state}

    if lora_state:
        print(f"[INFER][ckpt] Loading LoRA params into pipe.dit ({len(lora_state)} keys)")
        pipe.load_lora(pipe.dit, state_dict=lora_state)
    else:
        print("[INFER][ckpt] No LoRA keys found in checkpoint.")

    remapped_non_lora = {}
    for k, v in non_lora_state.items():
        if k.startswith("pipe."):
            inner = k[len("pipe."):]  
            remapped_non_lora[inner] = v
        else:
            continue

    if remapped_non_lora:
        print(f"[INFER][ckpt] Loading non-LoRA finetuned params into pipe ({len(remapped_non_lora)} keys)")
        pipe.load_state_dict(remapped_non_lora, strict=False)

def main():
    parser = argparse.ArgumentParser(description="Single-image inference for Qwen-Image-Edit-2509")
    parser.add_argument("--prompt", type=str, required=True, help="Edit prompt")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--save_path", type=str, required=True, help="Output image path")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/path/to/Qwen-Image-Edit-2509",
        help="Path to base Qwen-Image-Edit-2509 model",
    )
    parser.add_argument(
        "--dinov2_path",
        type=str,
        default="/path/to/DINOv2-with-registers-base",
        help="Path to DINOv2 model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to finetuned checkpoint (.safetensors)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Inference steps")
    args = parser.parse_args()

    pipe = QwenImagePhysicPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2509",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                local_model_path=args.base_model_path,
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
                local_model_path=args.base_model_path,
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                local_model_path=args.base_model_path,
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="tokenizer/",
            local_model_path=args.base_model_path,
        ),
        processor_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit",
            origin_file_pattern="processor/",
            local_model_path=args.base_model_path,
        ),
        dinov2_path=args.dinov2_path,
    )
    load_finetuned_into_pipe(pipe, args.lora_path)

    image = Image.open(args.image_path).convert("RGB")
    image = resize_image(image) 

    out = pipe(
        args.prompt,
        edit_image=image,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=image.size[1],
        width=image.size[0],
        is_train=False,
    )

    save_dir = os.path.dirname(os.path.abspath(args.save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    out.save(args.save_path)
    print(f"[DONE] Saved result to {args.save_path}")


if __name__ == "__main__":
    main()
