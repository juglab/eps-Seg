import pytorch_lightning as pl
from eps_seg.modules.lvae import LadderVAE
from eps_seg.config import LVAEConfig

class LVAEModel(pl.LightningModule):
    def __init__(self, config: LVAEConfig):
        super().__init__()
        self.cfg = config
        self.model = LadderVAE()

    def forward(self, inputs, labels=None):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_callbacks(self):
        return super().configure_callbacks()