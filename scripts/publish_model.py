import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def publish_model(
    checkpoint_dir: Path,
    repo_id: str,
    model_card_path: Path = Path("docs/model_card.md"),
) -> None:
    staging_dir = Path("/tmp/md-reheader-model-staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    # Copy only the files needed for inference
    for filename in [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]:
        src = checkpoint_dir / filename
        if src.exists():
            shutil.copy2(src, staging_dir / filename)

    # Copy model card as README.md
    shutil.copy2(model_card_path, staging_dir / "README.md")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Model uploaded to https://huggingface.co/{repo_id}")

    shutil.rmtree(staging_dir)


def publish_dataset(
    data_dir: Path,
    repo_id: str,
    dataset_card_path: Path = Path("docs/dataset_card.md"),
) -> None:
    staging_dir = Path("/tmp/md-reheader-dataset-staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    for split in ["train", "val", "test"]:
        src = data_dir / f"{split}.jsonl"
        if src.exists():
            shutil.copy2(src, staging_dir / f"{split}.jsonl")

    shutil.copy2(dataset_card_path, staging_dir / "README.md")

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")

    shutil.rmtree(staging_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_parser = subparsers.add_parser("model")
    model_parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints_v3/checkpoint-3082",
    )
    model_parser.add_argument(
        "--repo", type=str, default="joelbarmettler/md-reheader",
    )

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument(
        "--data-dir", type=str, default="./data/processed_v3",
    )
    dataset_parser.add_argument(
        "--repo", type=str, default="joelbarmettlerUZH/md-reheader-dataset",
    )

    args = parser.parse_args()

    if args.command == "model":
        publish_model(Path(args.checkpoint), args.repo)
    elif args.command == "dataset":
        publish_dataset(Path(args.data_dir), args.repo)


if __name__ == "__main__":
    main()
