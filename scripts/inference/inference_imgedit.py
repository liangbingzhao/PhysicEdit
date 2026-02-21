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
from PIL import Image
from diffsynth.pipelines.qwen_image_physical import QwenImagePhysicPipeline, ModelConfig
from diffsynth import load_state_dict
from pathlib import Path
from openai import OpenAI
import time
import json
from datasets import load_dataset
from typing import List, Dict, Any
from tqdm import tqdm
import math
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


EDIT_SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image’s overall scene.  

## 2. Task Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Do not translate or alter the original language of the text, and do not change the capitalization.  
- **For text replacement tasks, always use the fixed template:**
    - `Replace "xx" to "yy"`.  
    - `Replace the xx bounding box to "yy"`.  
- If the user does not specify text content, infer and add concise text based on the instruction and the input image’s context. For example:  
    > Original: "Add a line of text" (poster)  
    > Rewritten: "Add text \"LIMITED EDITION\" at the top center with slight shadow"  
- Specify text position, color, and layout in a concise way.  

### 3. Human Editing Tasks
- Maintain the person’s core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
- **For expression changes, they must be natural and subtle, never exaggerated.**  
- If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
    - For background change tasks, emphasize maintaining subject consistency at first.  
- Example:  
    > Original: "Change the person’s hat"  
    > Rewritten: "Replace the man’s hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"  

### 4. Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"  
- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them into the prompt.  
- **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"  
- If there are other changes, place the style description at the end.

## 3. Rationality and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.  
- Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).  

# Output Format Example
```json
{
   "Rewritten": "..."
}
'''

def polish_edit_prompt(edit_prompt: str) -> str:
    prompt = f"User Input: {edit_prompt} \n\n Rewritten Prompt:"
    success = False
    while not success:
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": EDIT_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            result = response.output_text
            if isinstance(result, str):
                result = result.replace('```json','')
                result = result.replace('```','')
                result = json.loads(result)
            else:
                result = json.loads(result)
            polished_prompt = result['Rewritten']
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            success = True
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            continue
    return polished_prompt

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_json(file_path: str) -> Dict[str, Any]:
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
    """
    从 train_new.py 保存的 ckpt 中，同时加载：
      1) DiT 的 LoRA 权重到 pipe.dit
      2) 新增模块（perceiver_resampler / visual_thinking_adapter 等）的普通权重到 pipe

    说明：
    - 训练时的模型是 QwenImageTrainingModule，参数名形如:
        * "pipe.dit.transformer_blocks.0.attn.to_q.lora_A.default.weight"
        * "pipe.perceiver_resampler.latents"
        * "pipe.visual_thinking_adapter.net.0.weight"
    - export_trainable_state_dict(remove_prefix="pipe.dit.") 之后，ckpt 里会变成:
        * DiT LoRA: "transformer_blocks.0.attn.to_q.lora_A.default.weight"
        * 新模块: 仍是 "pipe.perceiver_resampler.latents" / "pipe.visual_thinking_adapter..."

    推理时:
      - pipe.dit 上通过 load_lora 注入 LoRA（只看 lora_A / lora_B 的 key）
      - 其他以 "pipe." 开头的普通参数，去掉前缀 "pipe." 后 load_state_dict 到 pipe
    """
    if not ckpt_path:
        print("[INFER][ckpt] No checkpoint path provided, skip loading.")
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[INFER][ckpt] checkpoint not found: {ckpt_path}")

    print(f"[INFER][ckpt] Loading finetuned weights from {ckpt_path}")
    full_state = load_state_dict(ckpt_path)

    # 1) 拆分 LoRA 与非 LoRA
    lora_state = {k: v for k, v in full_state.items()
                  if ("lora_A" in k or "lora_B" in k)}
    non_lora_state = {k: v for k, v in full_state.items()
                      if k not in lora_state}

    # 2) 先加载 DiT 的 LoRA
    if lora_state:
        print(f"[INFER][ckpt] Loading LoRA params into pipe.dit ({len(lora_state)} keys)")
        pipe.load_lora(pipe.dit, state_dict=lora_state)
    else:
        print("[INFER][ckpt] No LoRA keys found in checkpoint.")

    # 3) 再加载新增模块的普通权重（perceiver_resampler / visual_thinking_adapter 等）
    #    这些 key 形如 "pipe.perceiver_resampler.xxx" -> "perceiver_resampler.xxx"
    remapped_non_lora = {}
    for k, v in non_lora_state.items():
        if k.startswith("pipe."):
            inner = k[len("pipe."):]  # 去掉 QwenImageTrainingModule 的前缀
            remapped_non_lora[inner] = v
        else:
            # 非 LoRA 非 pipe. 前缀的普通权重理论上很少，这里暂时忽略，避免误覆盖 base DiT
            continue

    if remapped_non_lora:
        print(f"[INFER][ckpt] Loading non-LoRA finetuned params into pipe ({len(remapped_non_lora)} keys)")
        # strict=False，且不打印 missing，以免刷屏；missing 大多是 base 模型参数，本来就不在 ckpt 里
        missing, unexpected = pipe.load_state_dict(remapped_non_lora, strict=False)


def main():
    parser = argparse.ArgumentParser(description="Inference with Qwen-Image-Edit-2509 + finetuned checkpoint")
    
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
        default="/path/to/ImgEdit_data",
        help="Root path to ImgEdit dataset (contains ImgEdit/ and ImgEdit_singleturndata/)"
    )
    # 为了兼容之前脚本，沿用参数名 lora_path，但现在表示的是"训练好的 checkpoint"
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/path/to/finetuned_checkpoint.safetensors",
        help="Path to finetuned checkpoint (.safetensors) saved by train_new.py"
    )

    # Inference parameters
    parser.add_argument("--output_path", type=str, default="/path/to/output", 
                       help="Output video path")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--height", type=int, default=480, 
                       help="Video height")
    parser.add_argument("--width", type=int, default=832, 
                       help="Video width")
    parser.add_argument("--num_inference_steps", type=int, default=40, 
                       help="Number of inference steps")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index of samples to process (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="End index of samples to process (exclusive)")
    parser.add_argument("--prompt_type", type=str, default="intermediate", choices=["intermediate", "explicit", "superficial"],help="Type of prompt to use")
    args = parser.parse_args()

    # load imgedit
    imgedit_data_root = os.path.join(args.data_path, "ImgEdit_singleturndata")
    imgedit_datainfo = load_json(os.path.join(args.data_path, "ImgEdit/Benchmark/Basic/basic_edit.json"))
    imgedit_data_items = list(imgedit_datainfo.items())
    
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
    
    out_root = Path(args.output_path).resolve()  # treat as DIR
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Determine processing range
    total_samples = len(imgedit_data_items)
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
        key, rec = imgedit_data_items[idx]
        
        # loading picabench specific data
        data_id = rec["id"]
        prompt = rec["prompt"]
        edit_type = rec["edit_type"]
        img_path = os.path.join(imgedit_data_root, data_id)

        input_image = Image.open(img_path)

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

        out_path = out_root / f"{key}.png"
        image.save(str(out_path))
        
        # Update progress bar description
        pbar.set_postfix({"current_idx": idx, "saved": out_path.name})

    print(f"[DONE] Generated {end_idx - start_idx} images into {out_root}")


if __name__ == "__main__":
    main() 