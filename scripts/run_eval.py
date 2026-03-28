import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/eval", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Evaluating with config: {cfg}")
    raise NotImplementedError("Evaluation pipeline not yet implemented. See Phase 4.")


if __name__ == "__main__":
    main()
