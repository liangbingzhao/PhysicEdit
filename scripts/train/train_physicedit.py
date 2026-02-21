import torch, os, json, time, shutil
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image_physical import QwenImagePhysicPipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, qwen_image_parser, launch_training_task, launch_data_process_task, PhysicalEditingDataset
from diffsynth.trainers.unified_dataset import UnifiedDataset
import wandb
from accelerate.utils import DistributedDataParallelKwargs
from pathlib import Path
import re
from typing import Optional, Dict, Any
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WandbModelLogger(ModelLogger):
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, use_wandb=False, save_every_n_steps=None, eval_every_n_steps=None, eval_data=None):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.use_wandb = use_wandb
        self.save_every_n_steps = save_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_data = eval_data
        if self.eval_data is not None:
            self.eval_data_iter = iter(self.eval_data)
        else:
            self.eval_data_iter = None
        self.global_step = 0
        
        # Check if we're on the main process
        import os
        self.is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
        
    def on_step_end(self, loss):
        self.global_step += 1
        
        # Save checkpoint every N steps if specified
        if self.save_every_n_steps is not None and self.global_step % self.save_every_n_steps == 0:
            # Note: We need accelerator and model to save, so we'll do this in the training loop
            pass
    
    def evaluate_model(self, model, accelerator):
        # breakpoint()
        if self.eval_data is None:
            return {}
        
        # Only evaluate on main process
        if not self.is_main_process:
            return {}

        # Get next sample from dataset
        try:
            eval_data = next(self.eval_data_iter)
        except StopIteration:
            self.eval_data_iter = iter(self.eval_data)
            eval_data = next(self.eval_data_iter)
        
        original_model = accelerator.unwrap_model(model)
        
        # Store original model state
        original_training_mode = original_model.training
        
        # Switch model to eval mode
        original_model.eval()
        
        with torch.no_grad():
            # Use the evaluation data (first sample from dataset) for inference
            # The eval_data contains: video, image, edit_image, prompt, etc.
            
            # Extract components from eval_data
            prompt = eval_data.get("prompt", "")
            gt_image = eval_data.get("image", None)
            edit_image = eval_data.get("edit_image", None)
            idx = eval_data.get("idx", None)
            transition = eval_data.get("transition", None)
            supported_rules = eval_data.get("supported_rules", None)
            contradicted_rules = eval_data.get("contradicted_rules", None)
            middle_key_frames = eval_data.get("middle_key_frames", None)
            stitched_image = eval_data.get("stitched_image", None)
            state = eval_data.get("state", None)
            
            try:
                image = original_model.pipe(
                    prompt=prompt,
                    seed=42,  # Fixed seed for consistent evaluation
                    height=480, width=832,
                    edit_image=edit_image, num_inference_steps=40,
                    edit_image_auto_resize=True,
                    is_train=False,
                )
                
                # Save image locally
                os.makedirs(self.output_path, exist_ok=True)
                image_path = os.path.join(self.output_path, f"eval_step_{self.global_step}_transition_{transition}_idx_{idx}.jpg")

                from PIL import Image

                def to_pil(img):
                    if isinstance(img, Image.Image):
                        return img.convert("RGB")
                    try:
                        import numpy as np
                        if isinstance(img, np.ndarray):
                            return Image.fromarray(img).convert("RGB")
                    except ImportError:
                        pass
                    if hasattr(img, "detach"):
                        img = img.detach().cpu()
                        import torch
                        if isinstance(img, torch.Tensor):
                            if img.dim() == 3:
                                if img.shape[0] in (1,3):
                                    img = img.permute(1, 2, 0)
                                img = img.numpy()
                            elif img.dim() == 4:
                                img = img.squeeze(0).permute(1,2,0).numpy()
                            return Image.fromarray((img * 255).astype("uint8")).convert("RGB")
                    return img

                if isinstance(edit_image, list):
                    edit_img = to_pil(edit_image[0])
                else:
                    edit_img = to_pil(edit_image)
                gt_img = to_pil(gt_image)
                out_img = to_pil(image)

                img_list = [img for img in [edit_img, gt_img, out_img] if img is not None]
                if len(img_list) < 3:
                    width, height = out_img.size if out_img is not None else (832, 480)
                    blank = Image.new("RGB", (width, height), (128, 128, 128))
                    while len(img_list) < 3:
                        img_list.append(blank)
                min_height = min(img.height for img in img_list)
                img_list = [img.resize((int(img.width * min_height / img.height), min_height), Image.BICUBIC) for img in img_list]

                total_width = sum(img.width for img in img_list)
                concat_img = Image.new("RGB", (total_width, min_height))
                x = 0
                for im in img_list:
                    concat_img.paste(im, (x, 0))
                    x += im.width
                concat_img.save(image_path)
                
                if self.use_wandb:
                    wandb.log({
                        "eval_step": self.global_step,
                        # "orignal_image": wandb.Image(edit_image),
                        # "gt_edit_image": wandb.Image(gt_image),
                        "eval_edit_image": wandb.Image(image_path),
                        "eval_prompt": prompt,
                    })
                
                print(f"[EVAL] Step {self.global_step}: Generated image saved to {image_path}")
                
                eval_metrics = {
                    "eval_step": self.global_step,
                    "eval_edit_image_path": image_path,
                }
                
            except Exception as e:
                print(f"[EVAL] Step {self.global_step}: Error during image generation: {e}")
                eval_metrics = {
                    "eval_step": self.global_step,
                    "eval_error": str(e),
                }
        
        original_model.pipe.scheduler.set_timesteps(1000, training=True)
        # Restore original training mode
        if original_training_mode:
            original_model.train()
        
        return eval_metrics
    
    def save_checkpoint(self, accelerator, model, step_id, is_epoch_end=False):
        accelerator.wait_for_everyone()
        if is_epoch_end:
            filename = f"epoch-{step_id}.safetensors"
        else:
            filename = f"step-{step_id}.safetensors"
        path = os.path.join(self.output_path, filename)

        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            accelerator.save(state_dict, path, safe_serialization=True)
            
        return path
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        self.save_checkpoint(accelerator, model, epoch_id, is_epoch_end=True)

class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        enable_fp8_training=False,
        task="sft",
        wandb_project=None,
        wandb_run_name=None,
        local_model_path=None,
        dinov2_path=None,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=enable_fp8_training, local_model_path=local_model_path, skip_download=local_model_path is not None)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/", local_model_path=local_model_path, skip_download=local_model_path is not None) if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/", local_model_path=local_model_path, skip_download=local_model_path is not None) if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePhysicPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config, dinov2_path=dinov2_path)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=enable_fp8_training,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.task = task

        # Initialize wandb only on main process
        if wandb_project is not None:
            # Check if we're on the main process
            import os
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "model_paths": model_paths,
                        "model_id_with_origin_paths": model_id_with_origin_paths,
                        "trainable_models": trainable_models,
                        "lora_base_model": lora_base_model,
                        "lora_target_modules": lora_target_modules,
                        "lora_rank": lora_rank,
                        "use_gradient_checkpointing": use_gradient_checkpointing,
                        "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
                        "extra_inputs": extra_inputs,
                    }
                )
                self.use_wandb = True
            else:
                self.use_wandb = False
        else:
            self.use_wandb = False

    
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
        }
        
        # Extra inputs
        controlnet_input, blockwise_controlnet_input = {}, {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("blockwise_controlnet_"):
                blockwise_controlnet_input[extra_input.replace("blockwise_controlnet_", "")] = data[extra_input]
            elif extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        if len(blockwise_controlnet_input) > 0:
            inputs_shared["blockwise_controlnet_inputs"] = [ControlNetInput(**blockwise_controlnet_input)]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            if unit is None:
                continue
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None, return_inputs=False, global_step=None):
        # Inputs
        # breakpoint()
        if inputs is None:
            inputs = self.forward_preprocess(data)
        else:
            inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        if return_inputs: return inputs
        
        # Loss
        if self.task == "sft":
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss = self.pipe.training_loss(global_step=global_step, **models, **inputs)
        elif self.task == "data_process":
            loss = inputs
        elif self.task == "direct_distill":
            loss = self.pipe.direct_distill_loss(**inputs)
        else:
            raise NotImplementedError(f"Unsupported task: {self.task}.")

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({
                "loss": loss.item(),
                "learning_rate": self.current_lr if hasattr(self, 'current_lr') else None,
                "special_token_loss": self.pipe.special_token_loss,
            })
        return loss

def _infer_step_number(name: str) -> int:
    match = re.search(r"(?:step|epoch)[-_](\d+)", name)
    return int(match.group(1)) if match else 0


def _is_accelerate_state_dir(path: Path) -> bool:
    required = ["optimizer_states.pt", "scheduler_states.pt", "random_states_0.pkl"]
    return any((path / item).exists() for item in required)


def _load_resume_metadata(target: Path, resume_type: str) -> Dict[str, Any]:
    if resume_type == "model":
        meta_path = target.with_suffix(".json")
    else:
        meta_path = target / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                if isinstance(metadata, dict):
                    return metadata
        except Exception as exc:
            print(f"read metadata failed: {meta_path}, error: {exc}")
    return {}


def _resolve_resume_target(resume_from: str, resume_type: str):
    target = Path(resume_from).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"resume_from path not exists: {target}")
    chosen_type = resume_type
    if target.is_dir():
        latest_json = target / "latest.json"
        if resume_type == "auto" and latest_json.exists():
            try:
                with open(latest_json, "r") as f:
                    latest_info = json.load(f)
                latest_name = latest_info.get("latest")
                if latest_name:
                    candidate = target / latest_name
                    if candidate.exists():
                        target = candidate
                        chosen_type = latest_info.get("type", resume_type)
            except Exception as exc:
                print(f"read {latest_json} failed: {exc}")
        if resume_type == "auto":
            chosen_type = "full" if _is_accelerate_state_dir(target) else "model"
        if chosen_type == "full":
            if not _is_accelerate_state_dir(target):
                raise ValueError(f"{target} is not an Accelerate state directory, cannot resume by full")
            metadata = _load_resume_metadata(target, "full")
            metadata.setdefault("global_step", _infer_step_number(target.name))
            metadata.setdefault("type", "full")
            metadata.setdefault("state_path", str(target))
            return chosen_type, target, metadata
        ckpts = sorted(target.glob("*.safetensors"), key=lambda p: (_infer_step_number(p.name), p.stat().st_mtime))
        if not ckpts:
            raise ValueError(f"{target} did not find safetensors checkpoint")
        latest = ckpts[-1]
        metadata = _load_resume_metadata(latest, "model")
        metadata.setdefault("global_step", _infer_step_number(latest.name))
        metadata.setdefault("type", "model")
        metadata.setdefault("checkpoint_path", str(latest))
        return "model", latest, metadata
    else:
        if resume_type == "auto":
            chosen_type = "model" if target.suffix == ".safetensors" else "full"
        metadata = _load_resume_metadata(target, chosen_type)
        parsed_name = target.name if chosen_type == "full" else target.stem
        metadata.setdefault("global_step", _infer_step_number(parsed_name))
        metadata.setdefault("type", chosen_type)
        if chosen_type == "model":
            metadata.setdefault("checkpoint_path", str(target))
        else:
            metadata.setdefault("state_path", str(target))
        return chosen_type, target, metadata


def _write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _save_checkpoint_metadata(ckpt_path: Path, metadata: Dict[str, Any]):
    if ckpt_path is None:
        return
    _write_json(ckpt_path.with_suffix(".json"), metadata)


if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    dataset = PhysicalEditingDataset(args=args)
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        enable_fp8_training=args.enable_fp8_training,
        task=args.task,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        local_model_path=args.local_model_path,
        dinov2_path=args.dinov2_path,
    )
    model_logger = WandbModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        use_wandb=model.use_wandb,
        save_every_n_steps=args.save_every_n_steps,
        eval_every_n_steps=args.eval_every_n_steps,
        eval_data=dataset if args.eval_every_n_steps is not None else None,
    )

    ################### custom wandb launcher ###################
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Custom training loop with wandb logging
    def custom_launch_training_task(dataset, model, model_logger, optimizer, scheduler, num_epochs, gradient_accumulation_steps):
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from accelerate import Accelerator

        # ==================== ADDED: 打印训练参数量 ====================
        print("\n" + "="*50)
        print("MODEL TRAINABLE PARAMETERS REPORT")
        print("="*50)
        total_trainable_params = 0
        trainable_param_names = []
        perceiver_resampler_params = 0
        visual_thinking_adapter_params = 0
        dino_resampler_params = 0
        dino_resampler_adapter_params = 0
        vae_resampler_params = 0
        vae_resampler_adapter_params = 0
        dino_time_embed_params = 0
        vae_time_embed_params = 0
        physical_transition_adapter_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_trainable_params += param.numel()
                trainable_param_names.append(name)
                if "perceiver" in name:
                    perceiver_resampler_params += param.numel()
                if "visual_thinking_adapter" in name:
                    visual_thinking_adapter_params += param.numel()
                if "dino_resampler" in name:
                    if "dino_resampler_adapter" in name:
                        dino_resampler_adapter_params += param.numel()
                    else:
                        dino_resampler_params += param.numel()
                if "vae_resampler" in name:
                    if "vae_resampler_adapter" in name:
                        vae_resampler_adapter_params += param.numel()
                    else:
                        vae_resampler_params += param.numel()
                if "dino_time_embed" in name:
                    dino_time_embed_params += param.numel()
                if "vae_time_embed" in name:
                    vae_time_embed_params += param.numel()
                if "physical_transition_adapter" in name:
                    physical_transition_adapter_params += param.numel()
        print(f"Total Trainable Params: {total_trainable_params / 1e6:.2f} Million")
        print(f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Million")
        print(f"Perceiver Resampler Params: {perceiver_resampler_params / 1e6:.2f} Million")
        print(f"Visual Thinking Adapter Params: {visual_thinking_adapter_params / 1e6:.2f} Million")
        print(f"DINO Resampler Params: {dino_resampler_params / 1e6:.2f} Million")
        print(f"DINO Resampler Adapter Params: {dino_resampler_adapter_params / 1e6:.2f} Million")
        print(f"VAE Resampler Params: {vae_resampler_params / 1e6:.2f} Million")
        print(f"VAE Resampler Adapter Params: {vae_resampler_adapter_params / 1e6:.2f} Million")
        print(f"DINO Time Embed Params: {dino_time_embed_params} ")
        print(f"VAE Time Embed Params: {vae_time_embed_params} ")
        print(f"Physical Transition Adapter Params: {physical_transition_adapter_params/1e6:.2f} Million")
        print("="*50 + "\n")
        ##########################################
        
        dataloader = DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers)
        try:
            total_batches_per_epoch = len(dataloader)
        except TypeError:
            total_batches_per_epoch = None
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)])

        ############################################
        resume_from = getattr(args, "resume_from", None)
        resume_type = getattr(args, "resume_type", "auto")
        resume_info = None
        resume_metadata: Dict[str, Any] = {}
        if resume_from:
            selected_type, target_path, metadata = _resolve_resume_target(resume_from, resume_type)
            resume_info = {"type": selected_type, "path": target_path}
            resume_metadata = metadata or {}
            if selected_type == "model":
                state_dict = load_state_dict(str(target_path))
                remove_prefix = args.remove_prefix_in_ckpt
                if remove_prefix:
                    # state_dict = {
                    #     (k if k.startswith(remove_prefix) else f"{remove_prefix}{k}"): v
                    #     for k, v in state_dict.items()
                    # }
                    fixed_state_dict = {}
                    for k, v in state_dict.items():
                        if not k.startswith("pipe."):
                            new_k = f"{remove_prefix}{k}"
                        else:
                            new_k = k
                        fixed_state_dict[new_k] = v
                    state_dict = fixed_state_dict

                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[RESUME][model] missing parameters: {missing}")
                if unexpected:
                    print(f"[RESUME][model] unexpected parameters: {unexpected}")
                print(f"[RESUME][model] resumed from {target_path}")

        ############################################
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

        local_steps_per_epoch = None
        try:
            local_steps_per_epoch = len(dataloader)
        except TypeError:
            pass
        if local_steps_per_epoch is not None:
            local_steps_per_epoch = int(local_steps_per_epoch)

        resume_global_step = int(resume_metadata.get("global_step", 0))
        resume_epoch_meta = int(resume_metadata.get("epoch", 0))
        original_num_processes = resume_metadata.get("num_processes") or getattr(args, "resume_original_num_processes", accelerator.num_processes)
        if original_num_processes is None:
            original_num_processes = accelerator.num_processes
        total_batches_per_epoch = resume_metadata.get("batches_per_epoch_total", total_batches_per_epoch)
        if total_batches_per_epoch not in (None, 0):
            total_batches_per_epoch = int(total_batches_per_epoch)
        total_batches_processed = resume_metadata.get("total_batches_processed")
        if total_batches_processed is None:
            total_batches_processed = resume_global_step * original_num_processes if original_num_processes is not None else resume_global_step
        else:
            total_batches_processed = int(total_batches_processed)

        resumed_using_full_state = False
        if resume_info and resume_info["type"] == "full":
            # now we don't save/load Accelerate state, disable full recovery, keep metadata for step inference
            print(f"[RESUME][full] disabled Accelerate state recovery, ignored {resume_info['path']}, continue by model weights/metadata inference")

        if total_batches_per_epoch is None or total_batches_per_epoch == 0:
            start_epoch = resume_epoch_meta if resumed_using_full_state else 0
            skipped_steps = 0
        else:
            if resumed_using_full_state:
                start_epoch = resume_epoch_meta
                skipped_steps = 0
            else:
                start_epoch = min(num_epochs, total_batches_processed // total_batches_per_epoch)
                remaining_total_batches = total_batches_processed - start_epoch * total_batches_per_epoch
                if local_steps_per_epoch:
                    skipped_steps = remaining_total_batches // max(1, accelerator.num_processes)
                    skipped_steps = min(skipped_steps, local_steps_per_epoch)
                else:
                    skipped_steps = 0

        if resume_info and accelerator.is_main_process:
            print(f"[RESUME] global_step={resume_global_step}, start_epoch={start_epoch}, skip_steps={skipped_steps}, full_state={resumed_using_full_state}")

        # Store reference to original model for accessing attributes
        original_model = accelerator.unwrap_model(model)

        if start_epoch >= num_epochs:
            print(f"[RESUME] {num_epochs} epochs planned have been completed (start_epoch={start_epoch}), end training")
            if original_model.use_wandb and model_logger.is_main_process:
                wandb.finish()
            return

        accelerator.wait_for_everyone()

        def build_metadata(save_type: str, step_id: int, epoch_id: int, current_global_step: int) -> Dict[str, Any]:
            metadata = {
                "global_step": current_global_step,
                "epoch": epoch_id,
                "step_id": step_id,
                "save_type": save_type,
                "num_processes": accelerator.num_processes,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "batches_per_epoch_total": total_batches_per_epoch,
                "local_steps_per_epoch": local_steps_per_epoch,
                "total_batches_processed": current_global_step * accelerator.num_processes,
                "timestamp": time.time(),
            }
            return metadata

        def save_training_snapshot(save_type: str, step_id: int, epoch_id: int, current_global_step: int, ckpt_path_str: Optional[str]):
            ckpt_path = Path(ckpt_path_str) if ckpt_path_str else None
            metadata = build_metadata(save_type, step_id, epoch_id, current_global_step)
            if ckpt_path is not None:
                metadata["checkpoint_path"] = str(ckpt_path)

            if accelerator.is_main_process and ckpt_path is not None:
                _save_checkpoint_metadata(ckpt_path, metadata)
            accelerator.wait_for_everyone()
            return metadata

        if resume_global_step > 0:
            model_logger.global_step = resume_global_step
            if hasattr(model_logger, "num_steps"):
                model_logger.num_steps = resume_global_step
            try:
                scheduler.last_epoch = resume_global_step
            except Exception:
                pass

        global_step = resume_global_step
        for epoch_id in range(start_epoch, num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")):
                if (not resumed_using_full_state) and epoch_id == start_epoch and resume_global_step > 0 and batch_idx < skipped_steps:
                    continue
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss = model(data, global_step=global_step)
                    accelerator.backward(loss)
                    optimizer.step()
                    model_logger.on_step_end(loss)
                    scheduler.step()
                    
                    # Update learning rate for logging
                    original_model.current_lr = scheduler.get_last_lr()[0]
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    global_step += 1
                    # breakpoint()
                    
                    # Save checkpoint every N steps if specified
                    if model_logger.save_every_n_steps is not None and global_step % model_logger.save_every_n_steps == 0:
                        ckpt_path = model_logger.save_checkpoint(accelerator, model, global_step, is_epoch_end=False)
                        save_training_snapshot("step", global_step, epoch_id, global_step, ckpt_path)
                    
                    # Evaluate model every N steps if specified
                    if model_logger.eval_every_n_steps is not None and global_step % model_logger.eval_every_n_steps == 0:
                        eval_results = model_logger.evaluate_model(model, accelerator)
                    
                    # Log to wandb every 10 steps (only on main process)
                    if original_model.use_wandb and model_logger.is_main_process and global_step % 10 == 0:
                        wandb.log({
                            "epoch": epoch_id,
                            "global_step": global_step,
                            "batch_loss": loss.item(),
                            "avg_epoch_loss": epoch_loss / num_batches,
                            "learning_rate": original_model.current_lr,
                        })
            
            # Log epoch summary (only on main process)
            if original_model.use_wandb and model_logger.is_main_process:
                wandb.log({
                    "epoch": epoch_id,
                    "epoch_loss": epoch_loss / num_batches,
                    "learning_rate": original_model.current_lr,
                })
            
            ckpt_path = model_logger.save_checkpoint(accelerator, model, epoch_id, is_epoch_end=True)
            save_training_snapshot("epoch", epoch_id, epoch_id, global_step, ckpt_path)
        
        # Close wandb (only on main process)
        if original_model.use_wandb and model_logger.is_main_process:
            wandb.finish()
    
    # Use custom training function instead of the default one
    custom_launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )