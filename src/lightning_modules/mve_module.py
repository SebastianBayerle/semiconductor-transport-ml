import torch
import torch.nn as nn
import pytorch_lightning as pl


class MVEModule(pl.LightningModule):
    """
    LightningModule for Mean-Variance Estimation.

    Expects model(x) -> mean, log_var

    Training:
        - warmup phase: MSE(mean, y)
        - main phase: Gaussian NLL(mean, log_var, y)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_epochs: int = 100,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def gaussian_nll_loss(self, mean, log_var, target):
        """
        Gaussian negative log likelihood, ignoring constant term.

        loss = 0.5 * (log(sigma^2) + (y - mean)^2 / sigma^2)
        """
        inv_var = torch.exp(-log_var)
        loss = 0.5 * (log_var + (target - mean) ** 2 * inv_var)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch

        mean, log_var = self(x)

        mse = self.mse_loss(mean, y)
        nll = self.gaussian_nll_loss(mean, log_var, y)

        if self.current_epoch < self.warmup_epochs:
            loss = mse
        else:
            loss = nll

        pred_std = torch.exp(0.5 * log_var)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mse", mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_nll", nll, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_mean_std", pred_std.mean(), prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        mean, log_var = self(x)

        val_mse = self.mse_loss(mean, y)
        val_nll = self.gaussian_nll_loss(mean, log_var, y)
        pred_std = torch.exp(0.5 * log_var)

        self.log("val_mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_nll", val_nll, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mean_std", pred_std.mean(), prog_bar=True, on_step=False, on_epoch=True)

        return val_nll

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
    def test_step(self, batch, batch_idx):
        x, y = batch
    
        mean, log_var = self(x)
    
        test_mse = self.mse_loss(mean, y)
        test_nll = self.gaussian_nll_loss(mean, log_var, y)
        pred_std = torch.exp(0.5 * log_var)
    
        self.log("test_mse", test_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_nll", test_nll, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_mean_std", pred_std.mean(), prog_bar=True, on_step=False, on_epoch=True)
    
        return test_nll
