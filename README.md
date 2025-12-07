<h1 align="center">OpenSubject</h1>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/AIPeanutman/OpenSubject"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow.svg" alt="Dataset"></a>
  <a href="https://huggingface.co/AIPeanutman/OpenSubject"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Model-blue.svg" alt="Model"></a>
  <a href="https://huggingface.co/datasets/AIPeanutman/OSBench"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Benchmark-green.svg" alt="Benchmark"></a>
</p>

<p align="center">
  <img src="assets/cover_fig.png" width="95%">
</p>

<p align="center">
OpenSubject is a video-derived large-scale corpus for subject-driven generation and manipulation.
</p>

## News 🚀🚀🚀

- **[2025-12]** OpenSubject released with OSBench evaluation benchmark
- **[2025-12]** Dataset and model weights available on Hugging Face

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/LAW1223/OpenSubject.git
```

### 2. Create a Clean Python Environment

```bash
conda create -n opensubject python=3.11
conda activate opensubject
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3.1 Install Flash Attention (Recommended)

```bash
# Note: Version 2.7.4.post1 is specified for compatibility with CUDA 12.4.
# Feel free to use a newer version if you use CUDA 12.6 or they fixed this compatibility issue.
# OmniGen2 runs even without flash-attn, though we recommend installing it for best performance.
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Dataset and Benchmark

### Download OpenSubject Dataset

Download the OpenSubject dataset from Hugging Face:

```bash
python scripts/hf_scripts/download_hf.py \
    --repo_id AIPeanutman/OpenSubject \
    --repo_type dataset \
    --local_dir /path/to/opensubject_dataset
```

### Download OSBench

Download the OSBench evaluation benchmark:

```bash
python scripts/hf_scripts/download_hf.py \
    --repo_id AIPeanutman/OSBench \
    --repo_type dataset \
    --local_dir /path/to/osbench
```

## Model Weights

### Download Pre-trained Model Weights

Download the base omnigen2 model weights:

```bash
python scripts/hf_scripts/download_hf.py \
    --repo_id OmniGen2/OmniGen2 \
    --repo_type model \
    --local_dir /path/to/omnigen2_model
```

Download the fine-tuned OpenSubject model weights:

```bash
python scripts/hf_scripts/download_hf.py \
    --repo_id AIPeanutman/OpenSubject \
    --repo_type model \
    --local_dir /path/to/opensubject_model
```

## Inference

### Quick Start with cli

The CLI tool (`scripts/inference_cli.py`) allows you to generate images directly from the command line.

#### Basic Usage

Generate an image from a text prompt:

```bash
python scripts/inference_cli.py \
    --model_path /path/to/omnigen2_model \
    --transformer_path /path/to/opensubject_model \
    --prompt "a beautiful landscape with mountains and lakes" \
    --output_path output.png \
    --num_inference_step 50 \
    --height 1024 \
    --width 1024
```

#### With Input Images (Image-to-Image)

Generate an image with reference input images:

```bash
python scripts/inference_cli.py \
    --model_path /path/to/omnigen2_model \
    --transformer_path /path/to/opensubject_model \
    --prompt "transform the scene to sunset" \
    --input_images input1.jpg input2.jpg \
    --output_path result.png \
    --num_inference_step 50
```

#### Key Parameters

- `--model_path`: Path to the model checkpoint (required)
- `--transformer_path`: Path to transformer checkpoint (optional, uses model_path/transformer if not specified)
- `--prompt`: Text prompt for image generation (required)
- `--input_images`: Input image paths or directory (optional, for image-to-image tasks)
- `--output_path`: Path to save generated image(s) (default: `output.png`)
- `--num_inference_step`: Number of inference steps (default: 50)
- `--height` / `--width`: Output image dimensions (default: 1024x1024)
- `--text_guidance_scale`: Text guidance scale (default: 5.0)
- `--image_guidance_scale`: Image guidance scale (default: 2.0)
- `--num_images_per_prompt`: Number of images to generate (default: 1)
- `--seed`: Random seed for reproducibility (default: 0)
- `--scheduler`: Scheduler type - `euler` or `dpmsolver` (default: `euler`)
- `--dtype`: Data type - `fp32`, `fp16`, or `bf16` (default: `bf16`)
- `--disable_align_res`: Disable resolution alignment to input images

## Evaluation

The evaluation pipeline consists of two steps:

1. **GPT-4 Based Scoring**: Uses GPT-4.1 to evaluate generated images
2. **Statistics Calculation**: Computes final metrics

### Quick Start with Provided Script

For convenience, we provide a complete inference and evaluation script at `scripts/inference.sh`. You can directly use this script by simply modifying the model and data paths:

```bash
# Edit the following variables in scripts/inference.sh:
# - model_path: Path to base OmniGen2 model
# - transformer_path: Path to OpenSubject fine-tuned transformer
# - test_data: Path to OSBench dataset
# - output_dir: Directory to save results
# - openai_key: Your OpenAI API key for evaluation

bash scripts/eval.sh
```

This script will automatically run inference and evaluation. For more detailed control, follow the manual steps below.

### Step1 Generate Images

Note: we fix the resolution of the output images at 720 × 1280 to ensure that the settings are consistent across different models.

You may try generating results using OmniGen2 or other models; please ensure that the output image directory structure and format are consistent with the format specified below.

```
results/
├── {method_name}/
│   └── fullset/
│       └── {task_type}/
│           ├── key1.png
│           ├── key2.png
│           └── ...
```

To use Opensubject, you can run the following script to generate images:

```bash

accelerate launch --num_processes=8 -m osbench.inference \
--model_path /path/to/omnigen2_model \
--transformer_path /path/to/opensubject_model \
--test_data /path/to/osbench \
--result_dir /path/to/results \
--num_inference_step 50 \
--height 720 \
--width 1280 \
--text_guidance_scale 5.0 \
--image_guidance_scale 2.0 \
--num_images_per_prompt 1 \
--disable_align_res 
```

###  Step2 Evaluation

1. We use GPT-4.1 to evaluate the quality of the generated images. Please make sure to set up your API key before running the script.

```bash
openai_key="<Your-API-Key>"

python -m osbench.test_osbench_score \
--test_data /path/to/osbench \
--result_dir /path/to/results \
--model_name "OmniGen2" \
--openai_key ${openai_key} \
--max_workers 16
```

2. Next, calculate the final score:

```bash
python -m osbench.calculate_statistics \
--save_path /path/to/results \
--model_name "OmniGen2" \
--backbone gpt4dot1
```

## References

Part of the code is based upon [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2). Thanks for their great work!

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{opensubject2025,
  title={OpenSubject: Subject-Driven Image Generation and Manipulation},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```