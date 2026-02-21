import torch
from PIL import Image
from typing import Union
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import List, Dict, Any
import json

from ..models import ModelManager, load_state_dict
from ..models.qwen_image_dit import QwenImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_image_vae import QwenImageVAE
from ..models.qwen_image_controlnet import QwenImageBlockWiseControlNet
from ..schedulers import FlowMatchScheduler
from ..utils import BasePipeline, ModelConfig, PipelineUnitRunner, PipelineUnit
from ..lora import GeneralLoRALoader
from .flux_image_new import ControlNetInput

from ..vram_management import gradient_checkpoint_forward, enable_vram_management, AutoWrappedModule, AutoWrappedLinear
import torch.nn as nn
from .helpers import PerceiverResampler, PhysicalTransitionAdapter, VisualThinkingAdapter, VisualThinkingDualAdapter, vt_get_alpha, VisualThinkingAdaLNAdapter

from .dinov2 import Dinov2withNorm
from torchvision import transforms

SPECIAL_TOKEN_NUM = 64

PHYSICAL_TRANSITIONS_MAP = {
    "no physics editing": 0,
    "active to dormant animal": 1,
    "dormant to active animal": 2,
    "decay": 3,
    "fruit ripening": 4,
    "germination": 5,
    "mature to flower": 6,
    "seedling to mature": 7,
    "disease to health": 8,
    "health to disease": 9,
    "mold growth": 10,
    "moss algae growth": 11,
    "alive to dead plant": 12,
    "intact to broken": 13,
    "hardening": 14,
    "softening": 15,
    "coating peeling": 16,
    "scractching abrasion": 17,
    "weathering": 18,
    "motion direction change": 19,
    "buoyancy": 20,
    "gravity": 21,
    "oscillation": 22,
    "rotation": 23,
    "translation": 24,
    "bending": 25,
    "collapse": 26,
    "compression": 27,
    "fracture": 28,
    "tension": 29,
    "torsion": 30,
    "light color change": 31,
    "light direction change": 32,
    "light intensity change": 33,
    "light temperature change": 34,
    "occlusion": 35,
    "light reflection": 36,
    "light refraction": 37,
    "clear to translucent": 38,
    "condensation": 39,
    "deposition": 40,
    "evaporation": 41,
    "freezing": 42,
    "melting": 43,
    "sublimation": 44,
    "cooling": 45,
    "heating": 46
}

SYSTEM_PROMPT_TRIPLET = """
You will transform ONE input text (prompt or caption) TOGETHER WITH:
(1) a chronological sequence of KEY FRAMES sampled from a video and
(2) two sets of physical principles: SUPPORTED_PRINCIPLES (aligned) and CONTRADICTED_PRINCIPLES (to avoid),
into TWO DETAILED prompt.

MAPPING FROM FRAMES TO OUTPUTS:
- middle_transition_prompt: use the key frames in temporal order, except the last frame, to describe the transition step-by-step. Favor mechanisms and cues that are consistent with SUPPORTED_PRINCIPLES. MUST be longer than the other prompt.
- final_state_prompt: describe exactly what is visible in the LAST key frame (final state).

EVIDENCE & PRECEDENCE:
- Primary: visual evidence in the key frames.
- Secondary: SUPPORTED_PRINCIPLES as positive constraints/mechanisms (use them only when consistent with what is visible).
- Tertiary: the input text (prompt/caption) as additional context if it does not conflict with what is visible.
- If any source conflicts with what is visible, FOLLOW THE FRAMES.
- STRICTLY AVOID describing mechanisms or outcomes that match CONTRADICTED_PRINCIPLES.

COMPLETENESS & OMISSION:
- You MAY supplement missing but VISIBLE details from the key frames (components, contacts, local changes).
- If a detail is neither visible nor clearly stated/entailed by SUPPORTED_PRINCIPLES, OMIT it; DO NOT GUESS.
- If a mechanism is suggested by SUPPORTED_PRINCIPLES but NOT visible, describe it only at a high level and
  only if it does not introduce new unobserved entities or attributes.

STRICT CONTENT RULES:
- No invention: do NOT introduce new objects, substances, tools, agents, causes, numbers, or attributes that are
  not visible in the frames or explicitly stated by the text or entailed by SUPPORTED_PRINCIPLES.
- ABSOLUTELY NO camera/filming/shot/lens/motion/angle/stabilization/exposure/ISO/focus/DoF/lighting-style words,
  and NO artistic style/medium descriptors (e.g., photorealistic, watercolor, cinematic, aesthetic).
- Do NOT mention Picture 1, Picture 2, Picture 3, etc. It should be a coherent description of the whole process.
- Use only entities, materials, colors, shapes, counts, spatial relations, contact interfaces, conditions, and
  sequence details that are visible/stated/entailed; never contradict the frames.

LEVEL OF DETAIL:
- All two fields must be richly detailed (multi-clause paragraphs). Include parts/components, spatial layout,
  relevant contacts/interfaces, and salient physical properties that are visible or safely entailed.
- The middle_transition_prompt should explain the mechanism of change using the intermediate frames:
  intermediate configurations, ordering, interactions (e.g., contact, mixing, dissolution, phase change,
  displacement, deformation), observable cues, and before→after indicators. Prefer cues supported by
  SUPPORTED_PRINCIPLES; avoid any content aligned with CONTRADICTED_PRINCIPLES.

FORMATTING:
- Return STRICT JSON ONLY:
{
  "middle_transition_prompt": "...",
  "final_state_prompt": "..."
}
- Each field: a paragraph (no bullet points), plain text, no quotes or code fences.
- Do NOT mention “prompt”, “caption”, “frame(s)”, “image(s)”, “principle(s)”, “camera”, “style”, or these instructions.

QUALITY CHECKS before returning:
- middle_transition_prompt is longer than final_state_prompt.
- No invented content; no camera/style words; no mention of frames/images/principles.
- No mechanisms or outcomes that match CONTRADICTED_PRINCIPLES.
""".strip()


SYSTEM_PROMPT_SAMPLE = """
You are a physics-aware visual editing assistant.
You will receive an "Edit Instruction" and an "Edit Image".
Your task is to generate a detailed description of the edit operations required to transform the image according to the instruction, ensuring all changes strictly follow physical laws.

INPUTS:
- Edit Instruction: The desired modification.
- Edit Image: The visual starting point.

REQUIREMENTS:
1. Physical Plausibility: All operations must respect physics (like gravity, inertia, material properties, light transport, collision, etc.).
2. Mechanism of Change: Describe *how* the change occurs visually (e.g., "The vase tilts and falls due to gravity," not just "The vase is on the floor").
3. Material Consistency: Ensure materials behave correctly (liquids flow, solids rigid/deform, cloth wrinkles).

OUTPUT FORMAT:
Return STRICT JSON ONLY:
{
  "middle_transition_prompt": "A multi-clause paragraph describing the step-by-step physical operations and visual transition."
}
""".strip()

class QwenImageBlockwiseMultiControlNet(torch.nn.Module):
    def __init__(self, models: list[QwenImageBlockWiseControlNet]):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = torch.nn.ModuleList(models)

    def preprocess(self, controlnet_inputs: list[ControlNetInput], conditionings: list[torch.Tensor], **kwargs):
        processed_conditionings = []
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            conditioning = rearrange(conditioning, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
            model_output = self.models[controlnet_input.controlnet_id].process_controlnet_conditioning(conditioning)
            processed_conditionings.append(model_output)
        return processed_conditionings

    def blockwise_forward(self, image, conditionings: list[torch.Tensor], controlnet_inputs: list[ControlNetInput], progress_id, num_inference_steps, block_id, **kwargs):
        res = 0
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            progress = (num_inference_steps - 1 - progress_id) / max(num_inference_steps - 1, 1)
            if progress > controlnet_input.start + (1e-4) or progress < controlnet_input.end - (1e-4):
                continue
            model_output = self.models[controlnet_input.controlnet_id].blockwise_forward(image, conditioning, block_id)
            res = res + model_output * controlnet_input.scale
        return res


class QwenImagePhysicPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, dinov2_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        from transformers import Qwen2Tokenizer, Qwen2VLProcessor
        
        self.scheduler = FlowMatchScheduler(sigma_min=0, sigma_max=1, extra_one_step=True, exponential_shift=True, exponential_shift_mu=0.8, shift_terminal=0.02)
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: QwenImageDiT = None
        self.vae: QwenImageVAE = None

        # DINO part
        assert dinov2_path is not None, "dinov2_path must be provided (path to DINOv2-with-registers-base)"
        self.dinov2 = Dinov2withNorm(dinov2_path=dinov2_path)
        self.dinov2.to(device=device, dtype=torch_dtype)
        proc_image_mean = [0.485, 0.456, 0.406]
        proc_image_std = [0.229, 0.224, 0.225]
        self.dinov2_mean = torch.tensor(proc_image_mean).view(1, 3, 1, 1)
        self.dinov2_std = torch.tensor(proc_image_std).view(1, 3, 1, 1)

        self.dino_resampler = PerceiverResampler(dim=768, num_latents=SPECIAL_TOKEN_NUM, depth=2)
        self.dino_resampler.to(device=device, dtype=torch_dtype)
        self.dino_time_embed = nn.Embedding(6, 768)
        self.dino_time_embed.to(device=device, dtype=torch_dtype)

        self.dino_resampler_adapter = VisualThinkingAdapter(in_dim=768, out_dim=3584)
        self.dino_resampler_adapter.to(device=device, dtype=torch_dtype)
        self.dino_input_size = 224

        # VAE part
        self.vae_resampler = PerceiverResampler(dim=64, num_latents=SPECIAL_TOKEN_NUM, depth=2, max_num_media_tokens=10240)
        self.vae_resampler.to(device=device, dtype=torch_dtype)
        self.vae_time_embed = nn.Embedding(6, 64)
        self.vae_time_embed.to(device=device, dtype=torch_dtype)
        self.vae_resampler_adapter = VisualThinkingAdapter(in_dim=64, out_dim=3584)
        self.vae_resampler_adapter.to(device=device, dtype=torch_dtype)

        # A dual head adapter
        self.visual_thinking_adapter = VisualThinkingDualAdapter(in_dim=3584, out_dim=3584, t_min=self.scheduler.timesteps.min().item(), t_max=self.scheduler.timesteps.max().item())
        self.visual_thinking_adapter.to(device=device, dtype=torch_dtype)


        self.blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None
        self.tokenizer: Qwen2Tokenizer = None
        self.processor: Qwen2VLProcessor = None
        self.unit_runner = PipelineUnitRunner()
        self.in_iteration_models = ("dit", "blockwise_controlnet", "visual_thinking_adapter")

        self.units = [
            QwenImageUnit_ShapeChecker(),
            QwenImageUnit_NoiseInitializer(),
            QwenImageUnit_InputImageEmbedder(),
            QwenImageUnit_Inpaint(),
            QwenImageUnit_EditImageEmbedder(),
            QwenImageUnit_ContextImageEmbedder(),
            QwenImageUnit_PhysicalVisualEmbedder(),
            QwenImageUnit_PhysicalVerbalEmbedder(),
            QwenImageUnit_PromptEmbedder(),
            QwenImageUnit_EntityControl(),
            QwenImageUnit_BlockwiseControlNet(),
        ]
        self.model_fn = model_fn_qwen_image

        
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)
            
            
    def clear_lora(self):
        for name, module in self.named_modules():
            if isinstance(module, AutoWrappedLinear): 
                if hasattr(module, "lora_A_weights"):
                    module.lora_A_weights.clear()
                if hasattr(module, "lora_B_weights"):
                    module.lora_B_weights.clear()
                    
    
    def enable_lora_magic(self):
        if self.dit is not None:
            if not (hasattr(self.dit, "vram_management_enabled") and self.dit.vram_management_enabled):
                dtype = next(iter(self.dit.parameters())).dtype
                enable_vram_management(
                    self.dit,
                    module_map = {
                        torch.nn.Linear: AutoWrappedLinear,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device=self.device,
                        onload_dtype=dtype,
                        onload_device=self.device,
                        computation_dtype=self.torch_dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=None,
                )
    
    def get_special_divisor(self, global_step=None, warmup_steps=10000):
        progress = min(1.0, global_step / float(warmup_steps))
        divisor = 10 - 9.0 * progress  
        return divisor
    
    def training_loss(self, global_step=None, **inputs):
        # breakpoint()
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        noise = torch.randn_like(inputs["input_latents"])
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], noise, timestep)
        
        noise_pred, special_token_loss = self.model_fn(**inputs, timestep=timestep)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # want to log into wandb
        self.special_token_loss = special_token_loss.detach().mean().item()
        loss = loss * self.scheduler.training_weight(timestep)
        loss += special_token_loss
        return loss
    
    
    def direct_distill_loss(self, **inputs):
        self.scheduler.set_timesteps(inputs["num_inference_steps"])
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred,_ = self.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
            inputs["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
        loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
        return loss
    
    
    def _enable_fp8_lora_training(self, dtype):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding, Qwen2RMSNorm, Qwen2_5_VisionPatchEmbed, Qwen2_5_VisionRotaryEmbedding
        from ..models.qwen_image_dit import RMSNorm
        from ..models.qwen_image_vae import QwenImageRMS_norm
        module_map = {
            RMSNorm: AutoWrappedModule,
            torch.nn.Linear: AutoWrappedLinear,
            torch.nn.Conv3d: AutoWrappedModule,
            torch.nn.Conv2d: AutoWrappedModule,
            torch.nn.Embedding: AutoWrappedModule,
            Qwen2_5_VLRotaryEmbedding: AutoWrappedModule,
            Qwen2RMSNorm: AutoWrappedModule,
            Qwen2_5_VisionPatchEmbed: AutoWrappedModule,
            Qwen2_5_VisionRotaryEmbedding: AutoWrappedModule,
            QwenImageRMS_norm: AutoWrappedModule,
        }
        model_config = dict(
            offload_dtype=dtype,
            offload_device="cuda",
            onload_dtype=dtype,
            onload_device="cuda",
            computation_dtype=self.torch_dtype,
            computation_device="cuda",
        )
        if self.text_encoder is not None:
            enable_vram_management(self.text_encoder, module_map=module_map, module_config=model_config)
        if self.dit is not None:
            enable_vram_management(self.dit, module_map=module_map, module_config=model_config)
        if self.vae is not None:
            enable_vram_management(self.vae, module_map=module_map, module_config=model_config)
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5, auto_offload=True, enable_dit_fp8_computation=False):
        self.vram_management_enabled = True
        if vram_limit is None and auto_offload:
            vram_limit = self.get_vram()
        if vram_limit is not None:
            vram_limit = vram_limit - vram_buffer
        
        if self.text_encoder is not None:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding, Qwen2RMSNorm, Qwen2_5_VisionPatchEmbed, Qwen2_5_VisionRotaryEmbedding
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    Qwen2_5_VLRotaryEmbedding: AutoWrappedModule,
                    Qwen2RMSNorm: AutoWrappedModule,
                    Qwen2_5_VisionPatchEmbed: AutoWrappedModule,
                    Qwen2_5_VisionRotaryEmbedding: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            from ..models.qwen_image_dit import RMSNorm
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            if not enable_dit_fp8_computation:
                enable_vram_management(
                    self.dit,
                    module_map = {
                        RMSNorm: AutoWrappedModule,
                        torch.nn.Linear: AutoWrappedLinear,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device="cpu",
                        onload_dtype=dtype,
                        onload_device=device,
                        computation_dtype=self.torch_dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=vram_limit,
                )
            else:
                enable_vram_management(
                    self.dit,
                    module_map = {
                        RMSNorm: AutoWrappedModule,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device="cpu",
                        onload_dtype=dtype,
                        onload_device=device,
                        computation_dtype=self.torch_dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=vram_limit,
                )
                enable_vram_management(
                    self.dit,
                    module_map = {
                        torch.nn.Linear: AutoWrappedLinear,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device="cpu",
                        onload_dtype=dtype,
                        onload_device=device,
                        computation_dtype=dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=vram_limit,
                )
        if self.vae is not None:
            from ..models.qwen_image_vae import QwenImageRMS_norm
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    QwenImageRMS_norm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.blockwise_controlnet is not None:
            enable_vram_management(
                self.blockwise_controlnet,
                module_map = {
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = None,
        dinov2_path: str = None,
    ):
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary()
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Initialize pipeline
        pipe = QwenImagePhysicPipeline(device=device, torch_dtype=torch_dtype, dinov2_path=dinov2_path)
        pipe.text_encoder = model_manager.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_manager.fetch_model("qwen_image_dit")
        pipe.vae = model_manager.fetch_model("qwen_image_vae")
        pipe.blockwise_controlnet = QwenImageBlockwiseMultiControlNet(model_manager.fetch_model("qwen_image_blockwise_controlnet", index="all"))
        if tokenizer_config is not None and pipe.text_encoder is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)

        # breakpoint()
        pipe.processor.tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_img>", "<end_of_img>"]
                    + [f"<img{i}>" for i in range(SPECIAL_TOKEN_NUM)]
                }
            )
        pipe.boi_token_id = pipe.processor.tokenizer.convert_tokens_to_ids("<begin_of_img>")
        pipe.eoi_token_id = pipe.processor.tokenizer.convert_tokens_to_ids("<end_of_img>")

        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Inpaint
        inpaint_mask: Image.Image = None,
        inpaint_blur_size: int = None,
        inpaint_blur_sigma: float = None,
        # Shape
        height: int = 1328,
        width: int = 1328,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        exponential_shift_mu: float = None,
        # Blockwise ControlNet
        blockwise_controlnet_inputs: list[ControlNetInput] = None,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        # Qwen-Image-Edit
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        edit_rope_interpolation: bool = False,
        # In-context control
        context_image: Image.Image = None,
        # FP8
        enable_fp8_attention: bool = False,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
        # Physical thinking
        supported_rules: list[Dict[str, Any]] = None,
        contradicted_rules: list[Dict[str, Any]] = None,
        middle_key_frames: list[Image.Image] = None,
        stitched_image: Image.Image = None,
        state: str = None,
        transition: str = None,
        triplet: dict = None,
        is_train: bool = True,
        have_text_reasoning: bool = True,
    ):
        # breakpoint()
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=(height // 16) * (width // 16), exponential_shift_mu=exponential_shift_mu)
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "inpaint_mask": inpaint_mask, "inpaint_blur_size": inpaint_blur_size, "inpaint_blur_sigma": inpaint_blur_sigma,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "enable_fp8_attention": enable_fp8_attention,
            "num_inference_steps": num_inference_steps,
            "blockwise_controlnet_inputs": blockwise_controlnet_inputs,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize, "edit_rope_interpolation": edit_rope_interpolation, 
            "context_image": context_image,
            "supported_rules": supported_rules,
            "contradicted_rules": contradicted_rules,
            "middle_key_frames": middle_key_frames,
            "stitched_image": stitched_image,
            "state": state,
            "transition": transition,
            "triplet": triplet,
            "is_train": is_train,
        }
        units = self.units
        if not is_train:
            units = [u for u in units if not isinstance(u, QwenImageUnit_PhysicalVisualEmbedder)]
        
        if not have_text_reasoning:
            units = [u for u in units if not isinstance(u, QwenImageUnit_PhysicalVerbalEmbedder)]

        for unit in units:
            # breakpoint()
            if unit is None:
                continue
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        # breakpoint()
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            # breakpoint()
            noise_pred_posi,_ = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, progress_id=progress_id)
            if cfg_scale != 1.0:
                noise_pred_nega,_ = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, progress_id=progress_id)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image



class QwenImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"))

    def process(self, pipe: QwenImagePhysicPipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class QwenImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed", "rand_device"))

    def process(self, pipe: QwenImagePhysicPipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}



class QwenImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePhysicPipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}



class QwenImageUnit_Inpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("inpaint_mask", "height", "width", "inpaint_blur_size", "inpaint_blur_sigma"),
        )

    def process(self, pipe: QwenImagePhysicPipeline, inpaint_mask, height, width, inpaint_blur_size, inpaint_blur_sigma):
        if inpaint_mask is None:
            return {}
        inpaint_mask = pipe.preprocess_image(inpaint_mask.convert("RGB").resize((width // 8, height // 8)), min_value=0, max_value=1)
        inpaint_mask = inpaint_mask.mean(dim=1, keepdim=True)
        if inpaint_blur_size is not None and inpaint_blur_sigma is not None:
            from torchvision.transforms import GaussianBlur
            blur = GaussianBlur(kernel_size=inpaint_blur_size * 2 + 1, sigma=inpaint_blur_sigma)
            inpaint_mask = blur(inpaint_mask)
        return {"inpaint_mask": inpaint_mask}


class QwenImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt","physical_txt": "physical_txt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            onload_model_names=("text_encoder",)
        )
        
    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        bool_mask = bool_mask[:, :hidden_states.shape[1]]
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def resize_image(self, image, target_area=384*384):
        width, height = self.calculate_dimensions(target_area, image.size[0] / image.size[1])
        return image.resize((width, height))
    
    def encode_prompt(self, pipe: QwenImagePhysicPipeline, prompt):
        template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.tokenizer(txt, max_length=4096+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
        if model_inputs.input_ids.shape[1] >= 1024:
            print(f"Warning!!! QwenImage model was trained on prompts up to 512 tokens. Current prompt requires {model_inputs['input_ids'].shape[1] - drop_idx} tokens, which may lead to unpredictable behavior.")
        hidden_states = pipe.text_encoder.edit_forward(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
        
    def encode_prompt_edit(self, pipe: QwenImagePhysicPipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

        # breakpoint()
        suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(SPECIAL_TOKEN_NUM)])
                + "<end_of_img><|im_end|>"
            )
        prompt_with_special_tokens = [p + suffix for p in prompt]
        drop_idx = 64
        txt = [template.format(e) for e in prompt_with_special_tokens]

        model_inputs = pipe.processor(text=txt, images=self.resize_image(edit_image), padding=True, return_tensors="pt").to(pipe.device)

        boi_pos = torch.where(model_inputs.input_ids == pipe.boi_token_id)[1]
        eoi_pos = torch.where(model_inputs.input_ids == pipe.eoi_token_id)[1]
        special_token_mask = torch.zeros_like(model_inputs.attention_mask, dtype=torch.bool)
        special_token_mask[:, boi_pos+1:eoi_pos] = True
        special_token_mask = special_token_mask[:, drop_idx:]

        hidden_states = pipe.text_encoder.edit_forward(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]

        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        return split_hidden_states, special_token_mask
    
    def encode_prompt_edit_multi(self, pipe: QwenImagePhysicPipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        base_img_prompt = "".join([img_prompt_template.format(i + 1) for i in range(len(edit_image))])
        txt = [template.format(base_img_prompt + e) for e in prompt]
        edit_image = [self.resize_image(image) for image in edit_image]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder.edit_forward(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states

    def process(self, pipe: QwenImagePhysicPipeline, prompt, edit_image=None, physical_txt=None, pseudo_special_emb=None) -> dict:
        # breakpoint()
        if physical_txt is not None:
            prompt += physical_txt
        special_token_mask = None
        if pipe.text_encoder is not None:
            prompt = [prompt]
            if edit_image is None:
                split_hidden_states = self.encode_prompt(pipe, prompt)
            elif isinstance(edit_image, Image.Image):
                split_hidden_states, special_token_mask = self.encode_prompt_edit(pipe, prompt, edit_image)
            else:
                split_hidden_states = self.encode_prompt_edit_multi(pipe, prompt, edit_image)
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask, "special_token_mask": special_token_mask}
        else:
            return {}

class QwenImageUnit_PhysicalVerbalEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image", "supported_rules", "contradicted_rules", "middle_key_frames", "input_image","triplet"),
            onload_model_names=("text_encoder", "tokenizer")
        )
        
    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def resize_image(self, image, target_area=384*384):
        width, height = self.calculate_dimensions(target_area, image.size[0] / image.size[1])
        return image.resize((width, height))

    def generate_text(self, pipe: QwenImagePhysicPipeline, model_inputs):
        decoded_ids = pipe.text_encoder.generate(**model_inputs, max_new_tokens=1000)
        decoded_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, decoded_ids)
        ]
        decoded_txt = pipe.tokenizer.batch_decode(decoded_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        try:
            filtered_txt = self._parse_generation_response(decoded_txt)
        except ValueError:
            return decoded_txt
        res = ""
        for k, v in filtered_txt.items():
            res += f"\n{k}: {v}"
        return res

    def _parse_generation_response(self, response: str) -> dict:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Cannot find JSON in response: {response}")

        payload = response[start:end + 1]
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse JSON: {payload}") from exc

        accepted_field_sets = (
            ("Reasoning",),
            ("physical_reasoning", "middle_transition_prompt", "final_state_prompt"),
            ("middle_transition_prompt",),
        )
        allowed = tuple({field for fields in accepted_field_sets for field in fields})
        result = {}
        for key in allowed:
            value = data.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"Field {key} must be string, got {type(value)}: {data}")
                result[key] = value.strip()

        present_keys = set(result.keys())
        if not any(present_keys == set(fields) for fields in accepted_field_sets):
            raise ValueError(
                f"Unsupported response format. Expected one of {accepted_field_sets}, got keys {sorted(present_keys)}: {data}"
            )

        return result
    
    def encode_physical_prompt_train(self, pipe: QwenImagePhysicPipeline, prompt, supported_rules, contradicted_rules, middle_key_frames, input_image):
        user_content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": "INPUT_TEXT:"},
            {"type": "input_text", "text": prompt},
            {"type": "input_text", "text": "SUPPORTED_PRINCIPLES:"},
            {"type": "input_text", "text": json.dumps(supported_rules, ensure_ascii=False)},
            {"type": "input_text", "text": "CONTRADICTED_PRINCIPLES:"},
            {"type": "input_text", "text": json.dumps(contradicted_rules, ensure_ascii=False)},
            {"type": "input_text", "text": "KEY_FRAMES_IN_ORDER:"},
        ]
        for _ in middle_key_frames:
            user_content.append({"type": "image"})  
        user_content.append({"type": "image"})
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TRIPLET
            },
            {
                "role": "user",
                "content": user_content
            },
        ]
        # breakpoint()
        messages = pipe.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        middle_key_frames = [self.resize_image(tmp_img) for tmp_img in middle_key_frames]
        middle_key_frames.append(self.resize_image(input_image))
        model_inputs = pipe.processor(text=[messages], images=middle_key_frames, padding=True, return_tensors="pt").to(pipe.device)

        decoded_txt = self.generate_text(pipe, model_inputs)
        # breakpoint()

        return decoded_txt

    def encode_physical_prompt_sample(self, pipe: QwenImagePhysicPipeline, edit_image, prompt):
        user_content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": "Edit Instruction:"},
            {"type": "input_text", "text": prompt},
            {"type": "input_text", "text": "Edit Image:"},
            {"type": "image"},
        ]
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_SAMPLE
            },
            {
                "role": "user",
                "content": user_content
            },
        ]
        # breakpoint()
        messages = pipe.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        model_inputs = pipe.processor(text=[messages], images=self.resize_image(edit_image), padding=True, return_tensors="pt").to(pipe.device)

        decoded_txt = self.generate_text(pipe, model_inputs)
        # breakpoint()

        return decoded_txt


    def process(self, pipe: QwenImagePhysicPipeline, prompt, edit_image=None, supported_rules=None, contradicted_rules=None, middle_key_frames=None, input_image=None, triplet=None) -> dict:
        # breakpoint()
        if pipe.text_encoder is not None:
            # pipe.load_models_to_device(self.onload_model_names)

            # encode physical prompt
            if supported_rules is not None and contradicted_rules is not None and middle_key_frames is not None and input_image is not None:
                # decoded_txt = self.encode_physical_prompt_train(pipe, prompt, supported_rules, contradicted_rules, middle_key_frames, input_image)
                
                # direct use gpt preprocess txt as decoded_txt
                # breakpoint()
                middle_transition_prompt = triplet.get("middle_transition_prompt", "")
                final_state_prompt = triplet.get("final_state_prompt", "")
                decoded_txt = f"Middle Transition Prompt: {middle_transition_prompt}\nFinal State Prompt: {final_state_prompt}"
            else:
                decoded_txt = self.encode_physical_prompt_sample(pipe, edit_image, prompt)
            return {"physical_txt": decoded_txt}
        else:
            return {}



class QwenImageUnit_PhysicalVisualEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("middle_key_frames", "edit_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=(
                "text_encoder",
                "dino_resampler",
                "dino_resampler_adapter",
                "vae_resampler",
                "vae_resampler_adapter",
            )
        )
        
    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def resize_image(self, image, target_area=384*384):
        width, height = self.calculate_dimensions(target_area, image.size[0] / image.size[1])
        return image.resize((width, height))

    def encode_middle_frame_desc(self, pipe: QwenImagePhysicPipeline, middle_key_frames):
        # breakpoint()
        template =  "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        base_img_prompt = "".join([img_prompt_template.format(i + 1) for i in range(len(middle_key_frames))])

        txt = [template.format(base_img_prompt)]
        middle_key_frames = [self.resize_image(image) for image in middle_key_frames]

        model_inputs = pipe.processor(text=txt, images=middle_key_frames, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder.edit_forward(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]

        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask[:,:hidden_states.shape[1]])
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        
        return split_hidden_states
    
    def dino_input_preprocess(self, pipe, middle_key_frames, dino_input_size):
        first_crop_size = int(dino_input_size * 1.5)
        transform = transforms.Compose(
            [
                transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(dino_input_size),
                transforms.ToTensor(),
            ]
        )
        middle_key_frames = [transform(image) for image in middle_key_frames]
        middle_tensor = torch.stack(middle_key_frames).to(pipe.device)
        middle_tensor = (middle_tensor - pipe.dinov2_mean.to(pipe.device)) / pipe.dinov2_std.to(pipe.device)
        return middle_tensor


    def process(self, pipe: QwenImagePhysicPipeline, middle_key_frames=None, edit_image=None, tiled=False, tile_size=None, tile_stride=None) -> dict:
        # breakpoint()
        if pipe.text_encoder is not None:
            # encode middle frame description
            split_hidden_states_middle = self.encode_middle_frame_desc(pipe, middle_key_frames)
            attn_mask_list_middle = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states_middle]
            max_seq_len = max([e.size(0) for e in split_hidden_states_middle])
            middle_frame_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states_middle])
            encoder_attention_mask_middle = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list_middle])
            middle_frame_embeds = middle_frame_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)

            # DINO + VAE dual visual embedder 
            ###### First calculate dino feature
            dino_inputs = self.dino_input_preprocess(pipe, middle_key_frames, pipe.dino_input_size)
            dino_hidden_states = pipe.dinov2(dino_inputs)

            # middle frame dino feature
            time_emb = pipe.dino_time_embed(torch.arange(len(middle_key_frames), device=dino_hidden_states.device))
            dino_middle_states = dino_hidden_states + time_emb.unsqueeze(1)
            dino_middle_states = rearrange(dino_middle_states, "B L H -> 1 (B L) H")
            dino_middle_resampler = pipe.dino_resampler(dino_middle_states)
            dino_middle_adapter = pipe.dino_resampler_adapter(dino_middle_resampler)

            # source image dino feature
            source_image_dino_inputs = self.dino_input_preprocess(pipe, [edit_image], pipe.dino_input_size)
            source_image_dino_hidden_states = pipe.dinov2(source_image_dino_inputs)
            source_image_dino_hidden_states = rearrange(source_image_dino_hidden_states, "B L H -> 1 (B L) H")
            source_image_dino_resampler = pipe.dino_resampler(source_image_dino_hidden_states)
            source_image_dino_adapter = pipe.dino_resampler_adapter(source_image_dino_resampler)

            # delta dino embedding
            pseudo_special_emb_dino = dino_middle_adapter - source_image_dino_adapter

            ###### Second calculate vae feature 
            processed_middle_frames = [pipe.preprocess_image(image).to(pipe.device, pipe.torch_dtype) for image in middle_key_frames]
            middle_latents = torch.cat([pipe.vae.encode(frame, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) for frame in processed_middle_frames])

            # Patchify Middle Latents
            middle_patched_latents = rearrange(middle_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=middle_latents.shape[2]//2, W=middle_latents.shape[3]//2, P=2, Q=2)
            # Add Time Embedding to Middle
            time_emb = pipe.vae_time_embed(torch.arange(len(middle_key_frames), device=middle_latents.device))
            middle_patched_latents = middle_patched_latents + time_emb.unsqueeze(1)
            # Reshape for Resampler
            middle_patched_latents = rearrange(middle_patched_latents, "B L H -> 1 (B L) H")

            vae_middle_resampler = pipe.vae_resampler(middle_patched_latents)
            vae_middle_emb = pipe.vae_resampler_adapter(vae_middle_resampler)

            processed_source_image = pipe.preprocess_image(edit_image).to(pipe.device, pipe.torch_dtype)
            source_latents = pipe.vae.encode(processed_source_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

            # Patchify Source Latents
            source_patched_latents = rearrange(source_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=source_latents.shape[2]//2, W=source_latents.shape[3]//2, P=2, Q=2)
            source_patched_latents = rearrange(source_patched_latents, "B L H -> 1 (B L) H")

            # delta vae embedding
            vae_source_resampler = pipe.vae_resampler(source_patched_latents)
            vae_source_emb = pipe.vae_resampler_adapter(vae_source_resampler)

            pseudo_special_emb_vae = vae_middle_emb - vae_source_emb

            return {"pseudo_special_emb_dino": pseudo_special_emb_dino, "pseudo_special_emb_vae": pseudo_special_emb_vae}
        else:
            return {}

class QwenImageUnit_EntityControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder",)
        )

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def get_prompt_emb(self, pipe: QwenImagePhysicPipeline, prompt) -> dict:
        if pipe.text_encoder is not None:
            prompt = [prompt]
            template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            drop_idx = 34
            txt = [template.format(e) for e in prompt]
            txt_tokens = pipe.tokenizer(txt, max_length=1024+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
            hidden_states = pipe.text_encoder.edit_forward(input_ids=txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True,)[-1]
            
            split_hidden_states = self.extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
        else:
            return {}

    def preprocess_masks(self, pipe, masks, height, width, dim):
        out_masks = []
        for mask in masks:
            mask = pipe.preprocess_image(mask.resize((width, height), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
            mask = mask.repeat(1, dim, 1, 1).to(device=pipe.device, dtype=pipe.torch_dtype)
            out_masks.append(mask)
        return out_masks

    def prepare_entity_inputs(self, pipe, entity_prompts, entity_masks, width, height):
        entity_masks = self.preprocess_masks(pipe, entity_masks, height//8, width//8, 1)
        entity_masks = torch.cat(entity_masks, dim=0).unsqueeze(0) # b, n_mask, c, h, w
        prompt_embs, prompt_emb_masks = [], []
        for entity_prompt in entity_prompts:
            prompt_emb_dict = self.get_prompt_emb(pipe, entity_prompt)
            prompt_embs.append(prompt_emb_dict['prompt_emb'])
            prompt_emb_masks.append(prompt_emb_dict['prompt_emb_mask'])
        return prompt_embs, prompt_emb_masks, entity_masks

    def prepare_eligen(self, pipe, prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, enable_eligen_on_negative, cfg_scale):
        entity_prompt_emb_posi, entity_prompt_emb_posi_mask, entity_masks_posi = self.prepare_entity_inputs(pipe, eligen_entity_prompts, eligen_entity_masks, width, height)
        if enable_eligen_on_negative and cfg_scale != 1.0:
            entity_prompt_emb_nega = [prompt_emb_nega['prompt_emb']] * len(entity_prompt_emb_posi)
            entity_prompt_emb_nega_mask = [prompt_emb_nega['prompt_emb_mask']] * len(entity_prompt_emb_posi)
            entity_masks_nega = entity_masks_posi
        else:
            entity_prompt_emb_nega, entity_prompt_emb_nega_mask, entity_masks_nega = None, None, None
        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb_posi, "entity_masks": entity_masks_posi, "entity_prompt_emb_mask": entity_prompt_emb_posi_mask}
        eligen_kwargs_nega = {"entity_prompt_emb": entity_prompt_emb_nega, "entity_masks": entity_masks_nega, "entity_prompt_emb_mask": entity_prompt_emb_nega_mask}
        return eligen_kwargs_posi, eligen_kwargs_nega

    def process(self, pipe: QwenImagePhysicPipeline, inputs_shared, inputs_posi, inputs_nega):
        eligen_entity_prompts, eligen_entity_masks = inputs_shared.get("eligen_entity_prompts", None), inputs_shared.get("eligen_entity_masks", None)
        if eligen_entity_prompts is None or eligen_entity_masks is None or len(eligen_entity_prompts) == 0 or len(eligen_entity_masks) == 0:
            return inputs_shared, inputs_posi, inputs_nega
        pipe.load_models_to_device(self.onload_model_names)
        eligen_enable_on_negative = inputs_shared.get("eligen_enable_on_negative", False)
        eligen_kwargs_posi, eligen_kwargs_nega = self.prepare_eligen(pipe, inputs_nega,
            eligen_entity_prompts, eligen_entity_masks, inputs_shared["width"], inputs_shared["height"],
            eligen_enable_on_negative, inputs_shared["cfg_scale"])
        inputs_posi.update(eligen_kwargs_posi)
        if inputs_shared.get("cfg_scale", 1.0) != 1.0:
            inputs_nega.update(eligen_kwargs_nega)
        return inputs_shared, inputs_posi, inputs_nega



class QwenImageUnit_BlockwiseControlNet(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("blockwise_controlnet_inputs", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def apply_controlnet_mask_on_latents(self, pipe, latents, mask):
        mask = (pipe.preprocess_image(mask) + 1) / 2
        mask = mask.mean(dim=1, keepdim=True)
        mask = 1 - torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        latents = torch.concat([latents, mask], dim=1)
        return latents

    def apply_controlnet_mask_on_image(self, pipe, image, mask):
        mask = mask.resize(image.size)
        mask = pipe.preprocess_image(mask).mean(dim=[0, 1]).cpu()
        image = np.array(image)
        image[mask > 0] = 0
        image = Image.fromarray(image)
        return image

    def process(self, pipe: QwenImagePhysicPipeline, blockwise_controlnet_inputs: list[ControlNetInput], tiled, tile_size, tile_stride):
        if blockwise_controlnet_inputs is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        conditionings = []
        for controlnet_input in blockwise_controlnet_inputs:
            image = controlnet_input.image
            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_image(pipe, image, controlnet_input.inpaint_mask)

            image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
            image = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_latents(pipe, image, controlnet_input.inpaint_mask)
            conditionings.append(image)
            
        return {"blockwise_controlnet_conditioning": conditionings}


class QwenImageUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "tiled", "tile_size", "tile_stride", "edit_image_auto_resize"),
            onload_model_names=("vae",)
        )


    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height


    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        return edit_image.resize((calculated_width, calculated_height))


    def process(self, pipe: QwenImagePhysicPipeline, edit_image, tiled, tile_size, tile_stride, edit_image_auto_resize=False):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(['vae'])
        if isinstance(edit_image, Image.Image):
            resized_edit_image = self.edit_image_auto_resize(edit_image) if edit_image_auto_resize else edit_image
            edit_image = pipe.preprocess_image(resized_edit_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            edit_latents = pipe.vae.encode(edit_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        else:
            resized_edit_image, edit_latents = [], []
            for image in edit_image:
                if edit_image_auto_resize:
                    image = self.edit_image_auto_resize(image)
                resized_edit_image.append(image)
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                edit_latents.append(latents)
        return {"edit_latents": edit_latents, "edit_image": resized_edit_image}


class QwenImageUnit_ContextImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("context_image", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePhysicPipeline, context_image, height, width, tiled, tile_size, tile_stride):
        if context_image is None:
            return {}
        pipe.load_models_to_device(['vae'])
        context_image = pipe.preprocess_image(context_image.resize((width, height))).to(device=pipe.device, dtype=pipe.torch_dtype)
        context_latents = pipe.vae.encode(context_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"context_latents": context_latents}


def model_fn_qwen_image(
    dit: QwenImageDiT = None,
    blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None,
    visual_thinking_adapter: VisualThinkingAdapter = None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    special_token_mask=None,
    height=None,
    width=None,
    blockwise_controlnet_conditioning=None,
    blockwise_controlnet_inputs=None,
    progress_id=0,
    num_inference_steps=1,
    entity_prompt_emb=None,
    entity_prompt_emb_mask=None,
    entity_masks=None,
    edit_latents=None,
    context_latents=None,
    enable_fp8_attention=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    edit_rope_interpolation=False,
    is_train=True,
    pseudo_special_emb_dino=None,
    pseudo_special_emb_vae=None,
    **kwargs
):
    # breakpoint()
    special_token_loss = 0
    if special_token_mask is not None:
        special_token = prompt_emb[special_token_mask].view(prompt_emb.shape[0], -1, prompt_emb.size(-1))
        special_token, dino_pred, vae_pred = visual_thinking_adapter(special_token, timestep)
        prompt_emb[special_token_mask] = special_token
        if is_train:
            special_token_loss = visual_thinking_adapter.get_loss(dino_pred, vae_pred, pseudo_special_emb_dino, pseudo_special_emb_vae, timestep)

    img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
    timestep = timestep / 1000
    
    image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
    image_seq_len = image.shape[1]

    if context_latents is not None:
        img_shapes += [(context_latents.shape[0], context_latents.shape[2]//2, context_latents.shape[3]//2)]
        context_image = rearrange(context_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=context_latents.shape[2]//2, W=context_latents.shape[3]//2, P=2, Q=2)
        image = torch.cat([image, context_image], dim=1)
    if edit_latents is not None:
        edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
        img_shapes += [(e.shape[0], e.shape[2]//2, e.shape[3]//2) for e in edit_latents_list]
        edit_image = [rearrange(e, "B C (H P) (W Q) -> B (H W) (C P Q)", H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2) for e in edit_latents_list]
        image = torch.cat([image] + edit_image, dim=1)

    image = dit.img_in(image)
    conditioning = dit.time_text_embed(timestep, image.dtype)

    if entity_prompt_emb is not None:
        text, image_rotary_emb, attention_mask = dit.process_entity_masks(
            latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask,
            entity_masks, height, width, image, img_shapes,
        )
    else:
        text = dit.txt_in(dit.txt_norm(prompt_emb))
        if edit_rope_interpolation:
            image_rotary_emb = dit.pos_embed.forward_sampling(img_shapes, txt_seq_lens, device=latents.device)
        else:
            image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        attention_mask = None
        
    if blockwise_controlnet_conditioning is not None:
        blockwise_controlnet_conditioning = blockwise_controlnet.preprocess(
            blockwise_controlnet_inputs, blockwise_controlnet_conditioning)

    for block_id, block in enumerate(dit.transformer_blocks):
        text, image = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            image=image,
            text=text,
            temb=conditioning,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
        )
        if blockwise_controlnet_conditioning is not None:
            image_slice = image[:, :image_seq_len].clone()
            controlnet_output = blockwise_controlnet.blockwise_forward(
                image=image_slice, conditionings=blockwise_controlnet_conditioning,
                controlnet_inputs=blockwise_controlnet_inputs, block_id=block_id,
                progress_id=progress_id, num_inference_steps=num_inference_steps,
            )
            image[:, :image_seq_len] = image_slice + controlnet_output
    
    image = dit.norm_out(image, conditioning)
    image = dit.proj_out(image)
    image = image[:, :image_seq_len]
    
    latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)
    return latents, special_token_loss
