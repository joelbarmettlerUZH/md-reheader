import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/training", config_name="full_ft_2gpu", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Training with config: {cfg}")
    raise NotImplementedError("Training loop not yet implemented. See Phase 3.")


if __name__ == "__main__":
    main()
