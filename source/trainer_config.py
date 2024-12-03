import questionary
from faim_ipa.utils import IPAConfig


class TrainerConfig(IPAConfig):
    max_epochs: int = 100
    precision: str = "32-true"

    def config_name(self) -> str:
        return "trainer_config.yaml"

    def prompt(self):
        self.max_epochs = int(
            questionary.text(
                "max_epochs",
                default=str(self.max_epochs),
            ).ask()
        )

        self.precision = questionary.select(
            "precision",
            choices=["32-true", "16-mixed"],
            default=self.precision,
        ).ask()

        self.save()


if __name__ == "__main__":
    config = TrainerConfig()
    config.prompt()

    config.save()
