import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from typing import List, Tuple
from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 CLI inference script.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to transformer checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--input_images",
        type=str,
        nargs="*",
        default=[],
        help="Paths to input images (can be multiple images or a directory).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save the generated image(s).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width."
    )
    parser.add_argument(
        "--max_input_image_pixels",
        type=int,
        default=1048576,
        help="Maximum number of pixels for each input image."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=5.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.0,
        help="Image guidance scale."
    )
    parser.add_argument(
        "--cfg_range_start",
        type=float,
        default=0.0,
        help="Start of the CFG range."
    )
    parser.add_argument(
        "--cfg_range_end",
        type=float,
        default=1.0,
        help="End of the CFG range."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    parser.add_argument(
        "--enable_group_offload",
        action="store_true",
        help="Enable group offload."
    )
    parser.add_argument(
        "--disable_align_res",
        action="store_true",
        help="Disable alignment to input image resolution."
    )
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    """Load the OmniGen2 pipeline."""
    from transformers import CLIPProcessor
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        processor=CLIPProcessor.from_pretrained(
            args.model_path,
            subfolder="processor",
            use_fast=True
        ),
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )

    if args.transformer_path:
        print(f"Transformer weights loaded from {args.transformer_path}")
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=weight_dtype,
        )
    else:
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    if args.scheduler == "dpmsolver":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(accelerator.device)
    return pipeline


def preprocess_input_images(input_image_paths: List[str]) -> List[Image.Image]:
    """Preprocess the input images."""
    input_images = []
    
    if not input_image_paths:
        return input_images
    
    # Handle directory input
    if len(input_image_paths) == 1 and os.path.isdir(input_image_paths[0]):
        image_dir = input_image_paths[0]
        image_files = [f for f in os.listdir(image_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        input_image_paths = [os.path.join(image_dir, f) for f in sorted(image_files)]
    
    # Load images
    for path in input_image_paths:
        if not os.path.exists(path):
            print(f"Warning: Image path {path} does not exist, skipping.")
            continue
        try:
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            input_images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}, skipping.")
    
    return input_images


def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    if not images:
        return None
    
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)


def main(args: argparse.Namespace) -> None:
    """Main function to run the CLI inference."""
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')
    
    if not accelerator.is_main_process:
        return
    
    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16
    
    # Load pipeline
    print("Loading pipeline...")
    pipeline = load_pipeline(args, accelerator, weight_dtype)
    print("Pipeline loaded successfully.")
    
    # Preprocess input images
    input_images = preprocess_input_images(args.input_images)
    if input_images:
        print(f"Loaded {len(input_images)} input image(s).")
    else:
        print("No input images provided, generating from text prompt only.")
    
    # Generate images
    print(f"Generating image(s) with prompt: {args.prompt}")
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    results = pipeline(
        prompt=args.prompt,
        input_images=input_images if input_images else None,
        width=args.width,
        height=args.height,
        align_res=not args.disable_align_res,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    
    # Save results
    output_dir = os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    if len(results.images) == 1:
        # Single image output
        results.images[0].save(args.output_path)
        print(f"Image saved to: {args.output_path}")
    else:
        # Multiple images output
        base_name, ext = os.path.splitext(args.output_path)
        for i, image in enumerate(results.images):
            output_path = f"{base_name}_{i}{ext}"
            image.save(output_path)
            print(f"Image {i+1} saved to: {output_path}")
        
        # Also save a collage if multiple images
        vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
        collage = create_collage(vis_images)
        collage_path = f"{base_name}_collage{ext}"
        collage.save(collage_path)
        print(f"Collage saved to: {collage_path}")
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)

