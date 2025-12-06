# OpenSubject

<p align="center">
  <img src="assets/cover_fig.pdf" width="95%">
</p>

OpenSubject is a comprehensive framework for subject-driven image generation and manipulation. Built upon OmniGen2, it provides a robust pipeline for training, evaluating, and deploying models capable of understanding and generating images with specific subjects across various contexts.

## News 🚀🚀🚀

- **[2025-12]** OpenSubject v1 released with OSBench evaluation benchmark
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

#### 3.1 Install PyTorch (choose correct CUDA version)

```bash
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
```

#### 3.2 Install Other Required Packages

```bash
pip install -r requirements.txt
```

#### 3.3 Install Flash Attention (Recommended)

```bash
# Note: Version 2.7.4.post1 is specified for compatibility with CUDA 12.4.
# Feel free to use a newer version if you use CUDA 12.6 or they fixed this compatibility issue.
# OmniGen2 runs even without flash-attn, though we recommend installing it for best performance.
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Dataset and Benchmark

### Download OpenSubject Dataset

Download the OpenSubject training dataset from Hugging Face:

```bash
python scripts/hf_scripts/download_hf.py \
    --repo_id AIPeanutman/OpenSubject \
    --repo_type dataset \
    --local_dir ./data/opensubject
```

### Download OSBench

Download the OSBench evaluation benchmark:

```bash
python scripts/hf_scripts/download_hf.py \
    --repo_id AIPeanutman/OSBench \
    --repo_type dataset \
    --local_dir ./data/osbench
```

For more details about the OSBench evaluation benchmark, please refer to [OSBench README](osbench/README.md).

## Model Weights

### Download Pre-trained Model Weights

Download the fine-tuned OpenSubject model weights:

```bash
python scripts/hf_scripts/download_hf.py \
    --repo_id AIPeanutman/OpenSubject \
    --repo_type model \
    --local_dir ./models/opensubject
```

The model weights will be downloaded to the specified local directory.

## Inference

### Quick Start

To generate images using the OpenSubject model:

```bash
bash scripts/inference.sh
```

You can modify the following parameters in `scripts/inference.sh`:

- `model_path`: Path to the base OmniGen2 model
- `transformer_path`: Path to the fine-tuned transformer weights
- `test_data`: Path to your test data
- `result_dir`: Directory to save generated results
- `num_inference_step`: Number of denoising steps (default: 50)
- `height` / `width`: Output image dimensions (default: 720x1280)
- `text_guidance_scale`: Text guidance scale (default: 5.0)
- `image_guidance_scale`: Image guidance scale (default: 2.0)

### Example Command

```bash
accelerate launch --num_processes=1 -m OSBench.inference \
    --model_path /path/to/OmniGen2 \
    --transformer_path /path/to/opensubject/transformer \
    --model_name "OmniGen2" \
    --test_data /path/to/test_data \
    --result_dir ./results \
    --num_inference_step 50 \
    --height 720 \
    --width 1280 \
    --text_guidance_scale 5.0 \
    --image_guidance_scale 2.0 \
    --num_images_per_prompt 1 \
    --dtype 'fp16' \
    --disable_align_res
```

## Evaluation

To evaluate the model on OSBench:

```bash
bash scripts/eval.sh
```

The evaluation pipeline consists of two steps:

1. **GPT-4 Based Scoring**: Uses GPT-4.1 to evaluate generated images
2. **Statistics Calculation**: Computes final metrics

For detailed evaluation instructions, please refer to [OSBench README](osbench/README.md).

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