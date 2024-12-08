import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from faim_ipa.utils import get_git_root

sys.path.append(str(get_git_root()))

from source.rdcnet.model import RDCNet2d
from source.data.BBBC039 import BBBC039
from source.rdcnet_config import RDCNetConfig
from source.trainer_config import TrainerConfig
from source.util import get_short_git_commit_hash


def main(
    rdcnet_config: RDCNetConfig,
    trainer_config: TrainerConfig,
):
    torch.set_float32_matmul_precision("high")
    dm = BBBC039(
        img_dir=get_git_root() / "raw_data" / "BBBC039" / "images",
        label_dir=get_git_root() / "raw_data" / "BBBC039" / "masks",
    )
    if rdcnet_config.model_path is not None:
        model = RDCNet2d.load_from_checkpoint(rdcnet_config.model_path)
        model.start_val_metrics_epoch = 0
    else:
        model = RDCNet2d(
            in_channels=rdcnet_config.in_channels,
            down_sampling_factor=rdcnet_config.down_sampling_factor,
            down_sampling_channels=rdcnet_config.down_sampling_channels,
            spatial_dropout_p=rdcnet_config.spatial_dropout_p,
            channels_per_group=rdcnet_config.channels_per_group,
            n_groups=rdcnet_config.n_groups,
            dilation_rates=rdcnet_config.dilation_rates,
            steps=rdcnet_config.steps,
            margin=rdcnet_config.margin,
        )

    output_dir = (
        get_git_root()
        / "processed_data"
        / f"{get_short_git_commit_hash()}_{Path(os.getcwd()).name}"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=trainer_config.max_epochs,
        precision=trainer_config.precision,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="mean_true_score",
                mode="max",
                save_top_k=1,
                filename="rdcnet-{epoch:02d}-{mean_true_score:.2f}",
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    v_num = trainer.logger.version

    results = trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path="best")

    with open(output_dir / f"{v_num}-test_results.yaml", "w") as f:
        yaml.safe_dump(results, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rdcnet_config",
        type=str,
        default="rdcnet_config.yaml",
    )
    parser.add_argument(
        "--trainer_config",
        type=str,
        default="trainer_config.yaml",
    )

    args = parser.parse_args()

    rdcnet_config = RDCNetConfig.load(Path(args.rdcnet_config))
    trainer_config = TrainerConfig.load(Path(args.trainer_config))

    main(
        rdcnet_config=rdcnet_config,
        trainer_config=trainer_config,
    )
