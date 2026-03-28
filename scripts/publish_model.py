from huggingface_hub import HfApi


def publish(
    checkpoint_dir: str = "./checkpoints/final",
    repo_id: str = "",
) -> None:
    if not repo_id:
        raise ValueError("Provide a repo_id, e.g. 'joelbarmettler/md-reheader'")

    api = HfApi()
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded model to {repo_id}")


if __name__ == "__main__":
    publish()
