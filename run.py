import os
import hydra
import logging
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

from duffing_system import system_training
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")

def run(cfg: DictConfig) -> None:
    print(f"Working directory: {os.getcwd()}")
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")
    logger.info(f"INFO LEVEL MESSAGE: \n")
    logger.info(f"CONFIG: \n{OmegaConf.to_yaml(cfg)}")
    system_training(cfg, output_dir, logger)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

