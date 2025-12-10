#!/usr/bin/env python3
"""
Extract OpenSubject image package files
Extract all tar.gz files to restore the original directory structure
"""

import os
import tarfile
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


class ImageExtractor:
    def __init__(self, packages_dir, output_dir, num_workers=8):
        """
        Initialize the extractor
        
        Args:
            packages_dir: Directory containing package files
            output_dir: Output directory for extraction (should contain Images directory)
            num_workers: Number of worker threads
        """
        self.packages_dir = Path(packages_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_packages(self):
        """Get all tar.gz files that need to be extracted"""
        packages = []
        
        # Match filename format: {task_type}_{image_type}_{subdir_name}.tar.gz
        pattern = re.compile(r'^(generation|manipulation)_(input_images|output_images)_(\d+)\.tar\.gz$')
        
        for package_file in sorted(self.packages_dir.glob('*.tar.gz')):
            match = pattern.match(package_file.name)
            if match:
                task_type, image_type, subdir_name = match.groups()
                packages.append({
                    'file': package_file,
                    'task_type': task_type,
                    'image_type': image_type,
                    'subdir_name': subdir_name
                })
        
        return packages
    
    def extract_package(self, package_info):
        """
        Extract a single package file
        
        Args:
            package_info: Dictionary containing package file information
            
        Returns:
            (success, message, file_count): Whether extraction succeeded, message, and file count
        """
        package_file = package_info['file']
        
        try:
            # Check if file exists
            if not package_file.exists():
                return False, f"File not found: {package_file.name}", 0
            
            # Open tar.gz file
            with tarfile.open(package_file, 'r:gz') as tar:
                # Get all file members
                members = tar.getmembers()
                file_count = len([m for m in members if m.isfile()])
                
                if file_count == 0:
                    return False, f"Empty package: {package_file.name}", 0
                
                # Check the path structure of the first file
                # Should be like: generation/input_images/0000/img_xxx.png
                first_file = next((m for m in members if m.isfile()), None)
                if first_file:
                    # Validate path structure
                    path_parts = Path(first_file.name).parts
                    if len(path_parts) < 3:
                        return False, f"Invalid path structure: {package_file.name}", 0
                    
                    # Validate that path starts with correct task_type and image_type
                    expected_task = package_info['task_type']
                    expected_image_type = package_info['image_type']
                    if path_parts[0] != expected_task or path_parts[1] != expected_image_type:
                        return False, f"Path mismatch: {package_file.name} (expected {expected_task}/{expected_image_type})", 0
                
                # Extract all files to output directory
                # Paths in tar are already relative (e.g., generation/input_images/0000/...)
                # Extract directly to output_dir, will automatically create complete directory structure
                tar.extractall(path=self.output_dir, members=members)
            
            return True, f"Success: {package_file.name}", file_count
        
        except tarfile.ReadError as e:
            return False, f"Read error: {package_file.name} - {str(e)}", 0
        except Exception as e:
            return False, f"Failed: {package_file.name} - {str(e)}", 0
    
    def extract_all(self):
        """Extract all package files"""
        # Get all package files
        print("Scanning package files...")
        packages = self.get_all_packages()
        total = len(packages)
        
        if total == 0:
            print("No package files found to extract!")
            print(f"Please ensure package files are in: {self.packages_dir}")
            return
        
        print(f"Found {total} package files to extract")
        print(f"Using {self.num_workers} worker threads")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        # Use thread pool for multi-threaded extraction
        success_count = 0
        fail_count = 0
        total_files = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_package = {
                executor.submit(self.extract_package, package_info): package_info
                for package_info in packages
            }
            
            # Use tqdm to show progress
            with tqdm(total=total, desc="Extraction progress", unit="packages", ncols=100) as pbar:
                for future in as_completed(future_to_package):
                    package_info = future_to_package[future]
                    try:
                        success, message, file_count = future.result()
                        total_files += file_count
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                            print(f"\nError: {message}")
                    except Exception as e:
                        fail_count += 1
                        print(f"\nException: {package_info['file'].name} - {str(e)}")
                    
                    pbar.update(1)
                    # Update progress bar description
                    pbar.set_postfix({
                        'Success': success_count,
                        'Failed': fail_count,
                        'Files': total_files
                    })
        
        # Print summary
        print("\n" + "=" * 60)
        print("Extraction completed!")
        print(f"Total packages: {total}")
        print(f"Successfully extracted: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Total files: {total_files:,}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nExtracted directory structure:")
        print(f"  {self.output_dir}/")
        print(f"    ├── generation/")
        print(f"    │   ├── input_images/")
        print(f"    │   └── output_images/")
        print(f"    └── manipulation/")
        print(f"        ├── input_images/")
        print(f"        └── output_images/")


def main():
    parser = argparse.ArgumentParser(description='Multi-threaded extraction of OpenSubject image package files')
    parser.add_argument(
        '--packages_dir',
        type=str,
        required=True,
        help='Directory containing package files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for extraction (default: Images directory in the same parent as packages_dir)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of worker threads (default: 8)'
    )
    
    args = parser.parse_args()
    
    # If output directory not specified, use default path
    if args.output_dir is None:
        packages_path = Path(args.packages_dir)
        args.output_dir = packages_path.parent / 'Images'
    
    # Create extractor and execute
    extractor = ImageExtractor(
        packages_dir=args.packages_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )
    
    extractor.extract_all()


if __name__ == '__main__':
    main()

