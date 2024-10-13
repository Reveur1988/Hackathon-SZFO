import os

import lightning
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import OmegaConf
from torch import set_float32_matmul_precision

from anomalib.data import get_datamodule
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import get_model
from src.callbacks.experiment_tracking import (
    ClearMLTracking,
)
from src.utils import PROJECT_ROOT


def train(cfg: dict):
    lightning.seed_everything(cfg["project_config"]["seed"])
    set_float32_matmul_precision("medium")

    # Initialize logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg["project_config"]["logs_dir"])

    # Initialize datamodule
    data_config = dict(cfg["data_config"])
    image_size = tuple(data_config["init_args"]["image_size"])
    data_config["init_args"]["image_size"] = image_size

    datamodule = get_datamodule(data_config)

    # Callbacks for training process
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            save_top_k=3,
            monitor="image_AUROC",  # f"pixel_{cfg.metrics['pixel'][0]}",
            mode="max",
            every_n_epochs=1,
        ),
    ]

    if cfg["project_config"]["track_in_clearml"]:
        tracking_cb = ClearMLTracking(cfg)
        callbacks += [
            tracking_cb,
        ]

    model = get_model(cfg["model"])  # , imagenet_dir="data/imagenette")

    engine = Engine(
        **dict(cfg["trainer_config"]),
        pixel_metrics=cfg["metrics"]["pixel"],
        task=data_config["init_args"]["task"],
        callbacks=callbacks,
        logger=tb_logger,
    )
    engine.fit(model=model, datamodule=datamodule)
    engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )
    engine.export(export_type=ExportType.TORCH, model=model, export_root="./")


if __name__ == "__main__":
    cfg_path = os.getenv("TRAIN_CFG_PATH", PROJECT_ROOT / "configs" / "train.yaml")

    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    train(cfg=cfg)
