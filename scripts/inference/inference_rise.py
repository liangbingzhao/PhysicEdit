import sys as _sys
import os as _os
# Ensure DiffSynth-Studio is on the Python path regardless of working directory.
_DIFFSYNTH_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../DiffSynth-Studio")
)
if _DIFFSYNTH_DIR not in _sys.path:
    _sys.path.insert(0, _DIFFSYNTH_DIR)
del _sys, _os, _DIFFSYNTH_DIR

import torch
import argparse
import os
import json
from PIL import Image
from diffsynth.pipelines.qwen_image_physical import QwenImagePhysicPipeline, ModelConfig
from diffsynth import load_state_dict
from pathlib import Path
from tqdm import tqdm


def load_json(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)

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
        missing, unexpected = pipe.load_state_dict(remapped_non_lora, strict=False)


def main():
    parser = argparse.ArgumentParser(description="Inference with Qwen-Image-Edit-2509 + finetuned checkpoint on RISEBench")
    
    # Model paths
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/path/to/Qwen-Image-Edit-2509",
        help="Path to base Qwen-Image-Edit-2509 model"
    )
    parser.add_argument(
        "--dinov2_path",
        type=str,
        default="/path/to/DINOv2-with-registers-base",
        help="Path to DINOv2 model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/RISEBench/data",
        help="Path to RISEBench data root"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/path/to/finetuned_checkpoint.safetensors",
        help="Path to finetuned checkpoint (.safetensors) saved by training script"
    )

    # Inference parameters
    parser.add_argument("--output_path", type=str, default="/path/to/output", 
                       help="Output path")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=40, 
                       help="Number of inference steps")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index of samples to process (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="End index of samples to process (exclusive)")
    args = parser.parse_args()

    # load RISEBench
    rise_datainfo = load_json(os.path.join(args.data_path, "datav2_total_w_subtask.json"))

    # Initialize pipeline
    print("Loading Qwen-Image-Edit-2509 pipeline...")
    pipe = QwenImagePhysicPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", local_model_path=args.base_model_path),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", local_model_path=args.base_model_path),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", local_model_path=args.base_model_path),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/", local_model_path=args.base_model_path),
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/", local_model_path=args.base_model_path),
        dinov2_path=args.dinov2_path,
    )
    
    print("Loading finetuned LoRA + new modules into pipeline...")
    load_finetuned_into_pipe(pipe, args.lora_path)

    # Enable VRAM management
    pipe.enable_vram_management()
    
    out_root = Path(args.output_path).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Determine processing range
    total_samples = len(rise_datainfo)
    if args.start_idx is None:
        start_idx = 0
    else:
        start_idx = args.start_idx
    
    if args.end_idx is None:
        end_idx = total_samples
    else:
        end_idx = min(args.end_idx, total_samples)
    
    print(f"Total samples: {total_samples}, Processing samples {start_idx} to {end_idx-1}")
    
    # Create progress bar
    pbar = tqdm(
        range(start_idx, end_idx),
        desc="Processing",
        total=end_idx - start_idx,
    )

    for idx in pbar:
        rec = rise_datainfo[idx]
        data_id = rec["index"]
        category = rec["category"]
        prompt = rec["instruction"]
        image_relpath = rec["image"]
        img_path = os.path.join(args.data_path, image_relpath)

        input_image = Image.open(img_path).convert("RGB")
        width, height = input_image.size

        image = pipe(
            prompt,
            edit_image=input_image,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            height=height,
            width=width,
            is_train=False,
        )

        out_dir = out_root / "images" / f"{category}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{data_id}.png"
        image.save(str(out_path))
        
        # Update progress bar description
        pbar.set_postfix({"current_idx": idx, "saved": out_path.name})

    print(f"[DONE] Generated {end_idx - start_idx} images into {out_root}")


if __name__ == "__main__":
    main()
