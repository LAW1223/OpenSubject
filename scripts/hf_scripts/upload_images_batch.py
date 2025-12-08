#!/usr/bin/env python3
"""
Batch upload Images_packages to HuggingFace Hub

This script uploads tar.gz files in batches to avoid memory issues.
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Batch upload Images_packages to HuggingFace")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/datasets/OpenSubject/Images_packages",
        help="Path to Images_packages directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="AIPeanutman/OpenSubject",
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of files to upload per batch"
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start from file index (for resuming)"
    )
    parser.add_argument(
        "--end_at",
        type=int,
        default=None,
        help="End at file index (for parallel uploads on multiple machines)"
    )
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HuggingFace token not provided")
        sys.exit(1)
    
    # Initialize API
    api = HfApi()
    
    # Get all tar.gz files
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Error: Directory {images_dir} not found")
        sys.exit(1)
    
    all_files = sorted(list(images_dir.glob("*.tar.gz")))
    total_files = len(all_files)
    print(f"Found {total_files} tar.gz files")
    
    # Apply start_from and end_at filters
    if args.end_at is not None:
        all_files = all_files[args.start_from:args.end_at]
        print(f"Uploading files {args.start_from} to {args.end_at-1} ({len(all_files)} files)")
    elif args.start_from > 0:
        all_files = all_files[args.start_from:]
        print(f"Starting from file index {args.start_from} ({len(all_files)} files)")
    
    print(f"Uploading to: {args.repo_id}/Images_packages/")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Upload files in batches
    failed_files = []
    
    for i in tqdm(range(0, len(all_files), args.batch_size), desc="Batches"):
        batch = all_files[i:i + args.batch_size]
        
        print(f"\nBatch {i//args.batch_size + 1}: Uploading {len(batch)} files...")
        
        for file_path in batch:
            try:
                print(f"  Uploading {file_path.name}...", end=" ")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"Images_packages/{file_path.name}",
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                failed_files.append(file_path.name)
    
    print()
    print("=" * 60)
    if failed_files:
        print(f"Upload completed with {len(failed_files)} failures:")
        for f in failed_files:
            print(f"  - {f}")
        print("\nYou can retry failed files by running the script again")
    else:
        print("✓ All files uploaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

