from dataclasses import dataclass

from grad_sdf.gui_base import GuiBase, GuiBaseConfig
from grad_sdf.trainer import Trainer, TrainerConfig


@dataclass
class GuiTrainerConfig(GuiBaseConfig):
    trainer: TrainerConfig = TrainerConfig()




class GuiTrainer(GuiBase):
    def __init__(self, cfg: GuiTrainerConfig):
        super().__init__(cfg)
        self.trainer = Trainer(cfg.trainer)
