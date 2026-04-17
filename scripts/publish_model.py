import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def _patch_for_transformers_v4_compat(staging_dir: Path) -> None:
    """Add legacy `rope_theta` alongside `rope_parameters` and drop the
    list-typed `extra_special_tokens` so vLLM (transformers 4.x) can load
    the model. transformers 5.x still reads the legacy fields fine."""
    cfg_path = staging_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        rp = cfg.get("rope_parameters")
        if rp and "rope_theta" not in cfg:
            cfg["rope_theta"] = rp.get("rope_theta", 1000000)
            cfg_path.write_text(json.dumps(cfg, indent=2))
            print("  patched config.json: added top-level rope_theta")

    tc_path = staging_dir / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        if isinstance(tc.get("extra_special_tokens"), list):
            tc.pop("extra_special_tokens")
            tc_path.write_text(json.dumps(tc, indent=2))
            print("  patched tokenizer_config.json: stripped extra_special_tokens list")


def publish_model(
    checkpoint_dir: Path,
    repo_id: str,
    model_card_path: Path = Path("docs/model_card.md"),
) -> None:
    staging_dir = Path("/tmp/md-reheader-model-staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

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

    _patch_for_transformers_v4_compat(staging_dir)

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
        "--repo", type=str, default="joelbarmettler/md-reheader-dataset",
    )

    args = parser.parse_args()

    if args.command == "model":
        publish_model(Path(args.checkpoint), args.repo)
    elif args.command == "dataset":
        publish_dataset(Path(args.data_dir), args.repo)


if __name__ == "__main__":
    main()
