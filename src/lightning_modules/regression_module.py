import torch
import torch.nn as nn
import pytorch_lightning as pl


class RegressionModule(pl.LightningModule):
    """
    LightningModule for standard supervised regression.

    Expects model(x) -> prediction
    Uses MSE loss.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
