import argparse
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload folder to HuggingFace Hub")
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Local path to the folder to upload"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'username/repo-name')"
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of the repo (default: model)"
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default=None,
        help="Path in the repo where to upload (e.g., 'transformer'). If None, uploads to root."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token. If not provided, will use token from environment or huggingface-cli login"
    )
    
    args = parser.parse_args()

api = HfApi()
    
    print(f"Uploading folder: {args.folder_path}")
    print(f"To repo: {args.repo_id}")
    print(f"Path in repo: {args.path_in_repo or 'root'}")
    print(f"Repo type: {args.repo_type}")

api.upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        token=args.token,
    )
    
    print("Upload completed successfully!")


if __name__ == "__main__":
    main()

'''
Example usage:

# Upload model transformer
python scripts/hf_scripts/upload_hf.py \
    --folder_path /path/to/transformer \
    --repo_id YourUsername/YourRepo \
    --repo_type model \
    --path_in_repo transformer \
    --token YOUR_HF_TOKEN

# Upload dataset
python scripts/hf_scripts/upload_hf.py \
    --folder_path /path/to/dataset \
    --repo_id YourUsername/YourDataset \
    --repo_type dataset \
    --token YOUR_HF_TOKEN
'''