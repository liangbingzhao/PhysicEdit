<div align="center" style="font-family: charter;">
<h1><i>From Statics to Dynamics:</i></br>Physics-Aware Image Editing with Latent Transition Priors</h1>


<p align="center">
  <a href="https://arxiv.org/abs/2602.21778"><img src="https://img.shields.io/badge/arXiv-PhysicEdit-b31b1b.svg" alt="arXiv"></a>
  <a href="https://liangbingzhao.github.io/statics2dynamics/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/collections/metazlb/physicedit-release"><img src="https://img.shields.io/badge/🤗-PhysicEdit-yellow" alt="HuggingFace"></a>
</p>


<div>
    <a href="https://liangbingzhao.github.io/" target="_blank">Liangbing Zhao</a><sup>1</sup>, </span>
    <a href="https://le-zhuo.com/" target="_blank">Le Zhuo</a><sup>2,3</sup>, </span>
    <a href="https://sayak.dev/" target="_blank">Sayak Paul</a><sup>4</sup>, </span>
    <a href="https://www.ee.cuhk.edu.hk/~hsli/" target="_blank">Hongsheng Li</a><sup>2</sup>, </span>
    <a href="https://cemse.kaust.edu.sa/profiles/mohamed-elhoseiny" target="_blank">Mohamed Elhoseiny</a><sup>1</sup></span>
</div>


<div>
    <sup>1</sup>KAUST&emsp;
    <sup>2</sup>CUHK MMLAB&emsp;
    <sup>3</sup>Krea AI&emsp;
    <sup>4</sup>Hugging Face&emsp;
</div>

<img src="figures/teaser.jpg" width="90%"/>

<p align="justify" style="font-size: 0.95em; max-width: 100%; margin: 0.4em auto 0;">
<b>Bridging semantic alignment and physical plausibility.</b>
Existing editing models achieve high semantic fidelity yet frequently violate physical principles, as they learn discrete image mappings with underspecified constraints.
We reformulate editing as a <b>Physical State Transition</b>, leveraging continuous dynamics to steer generation from <span style="color:red;">unreal hallucinations</span> toward <span style="color:green;">physically valid trajectories</span>.
</p>


</div> 

## 🔥 News

- **[2026/2/26]** — Release [paper](https://arxiv.org/abs/2602.21778).
- **[2026/2/25]** — Release PhysicTran38K dataset, model checkpoints, as well as the training and inference code.

## ✨ Quick Start 

### Installation 

```bash
git clone https://github.com/liangbingzhao/PhysicEdit.git
cd PhysicEdit

# Create and activate conda environment
conda create -n physicedit python=3.10 -y
conda activate physicedit

# Install DiffSynth environment
pip install -r DiffSynth-Studio/requirements.txt
pip install -e DiffSynth-Studio/
```

---

## 🚀 Model & Data Download

### Base Model

Download [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) and [DINOv2-with-registers-base](https://huggingface.co/facebook/dinov2-with-registers-base):

```bash
# Base model
python scripts/download_qwenimageedit.py --local_model_path /path/to/Qwen-Image-Edit-2509

# DINOv2
huggingface-cli download facebook/dinov2-with-registers-base --local-dir /path/to/DINOv2-with-registers-base
```

### Our model and dataset

| Name | Description | Link |
| --- | --- | --- |
| PhysicEdit | LoRA-finetuned checkpoint for physics-aware image editing | [HuggingFace](https://huggingface.co/metazlb/PhysicEdit) |
| PhysicTran38K | Video-based dataset of physical state transition | [HuggingFace](https://huggingface.co/datasets/metazlb/PhysicTran38K) |


## 🤖 Training

We provide training scripts in `scripts/train/`. You can start training with either single-GPU or multi-GPU configurations.

### Single-GPU Training

```bash
bash scripts/train/train_singlegpu.sh
```

### Multi-GPU Training

By default, the script uses 4 GPUs. You can override this by setting the `NUM_PROCESSES` environment variable.

```bash
# Default: 4 GPUs
bash scripts/train/train_multigpu.sh

# Override: 8 GPUs
NUM_PROCESSES=8 bash scripts/train/train_multigpu.sh
```

### Configuration

Before running, please update the following paths in the `.sh` files:

- `--dataset_base_path`: Path to your PhysicTran38K dataset
- `--local_model_path`: Path to Qwen-Image-Edit-2509
- `--dinov2_path`: Path to DINOv2-with-registers-base
- `--output_path`: Directory to save checkpoints and logs

### Advanced Options

You can enable **WandB logging** and **Resume Training** by uncommenting the corresponding lines at the end of the script.


## ⚡ Inference

After training, you can use `scripts/inference/validate.py` for simple inference on a single image:

```bash
python scripts/inference/validate.py \
    --prompt input_prompt \
    --image_path /path/to/input.jpg \
    --save_path /path/to/output.jpg \
    --base_model_path /path/to/Qwen-Image-Edit-2509 \
    --dinov2_path /path/to/DINOv2-with-registers-base \
    --lora_path /path/to/physicedit_checkpoint.safetensors
```

We also provide inference scripts for image editing benchmarks in `scripts/inference/`, including **PICABench** and **KRIS-Bench**. You can refer to the respective scripts (`inference_*.py`) for usage details.

## 🤝 Acknowledgement

This codebase is heavily built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). We also thank [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [RAE](https://github.com/bytetriper/RAE), and [flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch) for their excellent open-source work.


## ✏️ Citation

If you find this work for your research and applications, please cite using this BibTeX:

```bibtex
@article{[CITE_KEY],
  title   = {[PAPER TITLE]},
  author  = {[AUTHORS]},
  journal = {[VENUE]},
  year    = {[YEAR]},
  url     = {[ARXIV_LINK]}
}
```

