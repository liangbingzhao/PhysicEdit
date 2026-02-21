#!/bin/bash
# Training script for Qwen-Image-Edit-2509 physical editing.
# Run from any directory; DIFFSYNTH_DIR is resolved relative to this script's location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFSYNTH_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/DiffSynth-Studio"
# Default to 4 GPUs, override with env var, e.g. NUM_PROCESSES=8 bash train_multigpu.sh
NUM_PROCESSES="${NUM_PROCESSES:-4}"

cd "$DIFFSYNTH_DIR" || { echo "ERROR: cannot cd to $DIFFSYNTH_DIR"; exit 1; }

PYTHONPATH="$DIFFSYNTH_DIR:$PYTHONPATH" \
accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" "$SCRIPT_DIR/train_physicedit.py" \
  --dataset_base_path path/to/our_dataset \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --data_file_keys "image" \
  --extra_inputs "edit_image,supported_rules,contradicted_rules,middle_key_frames,stitched_image,state,transition,triplet" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --local_model_path path/to/original_Qwen-Image-Edit-2509_ckpt \
  --dinov2_path path/to/DINOv2-with-registers-base \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path path/to/output \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 128 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_every_n_steps 8000 \
  --eval_every_n_steps 4000 \
  --trainable_models "visual_thinking_adapter,vae_time_embed,vae_resampler,vae_resampler_adapter,dino_time_embed,dino_resampler,dino_resampler_adapter"
  # --wandb_project "your_wandb_project_name" \
  # --wandb_run_name "your_wandb_run_name" \
  # --resume_from path/to/your/checkpoint/epoch-0.safetensors \
  # --resume_original_num_processes "$NUM_PROCESSES" \
