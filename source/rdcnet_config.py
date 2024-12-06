from pathlib import Path
from typing import Optional

import questionary
from faim_ipa.utils import IPAConfig


class RDCNetConfig(IPAConfig):
    model_path: Optional[Path] = None
    in_channels: int = 1
    down_sampling_factor: int = 6
    down_sampling_channels: int = 8
    spatial_dropout_p: float = 0.1
    channels_per_group: int = 32
    n_groups: int = 4
    dilation_rates: list[int] = [1, 2, 4, 8, 16]
    steps: int = 6
    margin: int = 10
    lr: float = 0.001
    min_votes_per_instance: int = 5

    def config_name(self) -> str:
        return "rdcnet_config.yaml"

    def prompt(self):
        fine_tune = questionary.confirm("Fine tune?").ask()
        if fine_tune:
            self.model_path = questionary.path(
                "Model path",
                default=str(self.model_path),
            ).ask()
        else:
            self.in_channels = int(
                questionary.text(
                    "in_channels",
                    default=str(self.in_channels),
                ).ask()
            )
            self.down_sampling_factor = int(
                questionary.text(
                    "down_sampling_factor",
                    default=str(self.down_sampling_factor),
                ).ask()
            )
            self.down_sampling_channels = int(
                questionary.text(
                    "down_sampling_channels",
                    default=str(self.down_sampling_channels),
                ).ask()
            )
            self.spatial_dropout_p = float(
                questionary.text(
                    "spatial_dropout_p",
                    default=str(self.spatial_dropout_p),
                ).ask()
            )
            self.channels_per_group = int(
                questionary.text(
                    "channels_per_group",
                    default=str(self.channels_per_group),
                ).ask()
            )
            self.n_groups = int(
                questionary.text(
                    "n_groups",
                    default=str(self.n_groups),
                ).ask()
            )
            self.dilation_rates = list(
                map(
                    int,
                    questionary.text(
                        "dilation_rates",
                        default=",".join(map(str, self.dilation_rates)),
                    )
                    .ask()
                    .split(","),
                )
            )
            self.steps = int(
                questionary.text(
                    "steps",
                    default=str(self.steps),
                ).ask()
            )
            self.margin = int(
                questionary.text(
                    "margin",
                    default=str(self.margin),
                ).ask()
            )
            self.lr = float(
                questionary.text(
                    "lr",
                    default=str(self.lr),
                ).ask()
            )
            self.min_votes_per_instance = int(
                questionary.text(
                    "min_votes_per_instance",
                    default=str(self.min_votes_per_instance),
                ).ask()
            )

        self.save()


if __name__ == "__main__":
    config = RDCNetConfig()
    config.prompt()
