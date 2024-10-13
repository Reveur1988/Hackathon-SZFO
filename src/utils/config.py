from pathlib import Path
from typing import Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel


class _BaseValidatedConfig(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


class ProjectConfig(_BaseValidatedConfig):
    project_name: str = "default_project_name"
    experiment_name: str = "default_experiment_name"
    track_in_clearml: bool = True
    seed: int = 13
    logs_dir: str = "logs"


class DataConfig(_BaseValidatedConfig):
    name: str = "folder"  # options: str [mvtec, btech, folder]
    format: str = "mvtec"
    path: str = "./data"
    task: str = "detection"
    category: str = "bottle"
    image_size: str = 512
    train_batch_size: str = 1
    test_batch_size: str = 1
    num_workers: str = 8
    transform_config: dict = dict(
        train=None,
        val=None,
    )
    create_validation_set: bool = False
    tiling: dict = dict(
        apply=False,
        tile_size=None,
        stride=None,
        remove_border_count=0,
        use_random_tiling=False,
        random_tile_count=16,
    )


class TrainerConfig(_BaseValidatedConfig):
    accelerator: str = "gpu"  # <cpu, gpu, tpu, ipu, hpu, auto>
    benchmark: bool = False
    check_val_every_n_epoch: int = 1  # Don't validate before extracting features.
    default_root_dir: str = "logs"
    detect_anomaly: bool = False
    deterministic: bool = False
    devices: int = 1
    enable_checkpointing: bool = True
    enable_model_summary: bool = True
    enable_progress_bar: bool = True
    fast_dev_run: bool = False
    gradient_clip_val: int = 0
    limit_predict_batches: float = 1.0
    limit_test_batches: float = 1.0
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    log_every_n_steps: int = 50
    max_epochs: int = 5
    max_steps: int = -1
    max_time: int = None
    min_epochs: int = 4
    min_steps: int = None
    num_nodes: int = 1
    num_sanity_val_steps: int = 0
    overfit_batches: float = 0.0
    plugins: str = None
    precision: int = 32
    profiler: str = None
    reload_dataloaders_every_n_epochs: int = 0
    replace_sampler_ddp: bool = True
    strategy: str = "auto"
    sync_batchnorm: bool = False
    val_check_interval: float = 1.0  # Don't validate before extracting features.


class OptimizerConfig(_BaseValidatedConfig):
    name: str = "SGD"
    lr: float = 2e-3
    weight_decay: float = 0.01


class SchedulerConfig(_BaseValidatedConfig):
    warmup_steps: int = 200
    num_cycles: int = 2


class ExperimentConfig:
    # project_config: ProjectConfig = Field(default=ProjectConfig())
    # data_config: DataConfig = Field(default=DataConfig())

    # model: dict
    # trainer_config: TrainerConfig = Field(default=TrainerConfig())
    # metrics: dict

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]):
        with open(path, "w") as out_file:
            yaml.safe_dump(
                self.model_dump(),
                out_file,
                default_flow_style=False,
                sort_keys=False,
            )
