import torch
import torch.nn as nn
import pytorch_lightning as pl



class MVEModule(pl.LightningModule):
    """
    LightningModule for Mean-Variance Estimation with staged training.

    Expected model:
        model(x) -> mean, log_var

    Training phases:
        1. mean phase:
            train shared + mean_head
            freeze log_var_head
            loss = MSE(mean, y)

        2. variance phase:
            freeze shared + mean_head
            train log_var_head
            loss = Gaussian NLL(mean.detach(), log_var, y)

        3. optional joint phase:
            train all
            loss = Gaussian NLL(mean, log_var, y)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        mean_weight_decay: float = 1e-5,
        var_weight_decay: float = 1e-4,
        mean_epochs: int = 250,
        variance_epochs: int = 250,
        joint_training: bool = False,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.mean_weight_decay = mean_weight_decay
        self.var_weight_decay = var_weight_decay
        self.mean_epochs = mean_epochs
        self.variance_epochs = variance_epochs
        self.joint_training = joint_training

        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def gaussian_nll_loss(self, mean, log_var, target):
        """
        Gaussian negative log-likelihood without constant term.

        loss = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
        """
        inv_var = torch.exp(-log_var)
        loss = 0.5 * (log_var + (target - mean) ** 2 * inv_var)
        return loss.mean()

    def current_phase(self):
        if self.current_epoch < self.mean_epochs:
            return "mean"

        if self.current_epoch < self.mean_epochs + self.variance_epochs:
            return "variance"

        if self.joint_training:
            return "joint"

        return "variance"

    def _set_requires_grad(self, module, requires_grad):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _set_training_phase(self):
        phase = self.current_phase()

        if phase == "mean":
            # Train shared representation and mean head only
            self._set_requires_grad(self.model.shared, True)
            self._set_requires_grad(self.model.mean_head, True)
            self._set_requires_grad(self.model.log_var_head, False)

        elif phase == "variance":
            # Freeze mean predictor completely
            self._set_requires_grad(self.model.shared, False)
            self._set_requires_grad(self.model.mean_head, False)
            self._set_requires_grad(self.model.log_var_head, True)

        elif phase == "joint":
            # Optional NLL fine-tuning of all parts
            self._set_requires_grad(self.model.shared, True)
            self._set_requires_grad(self.model.mean_head, True)
            self._set_requires_grad(self.model.log_var_head, True)

        else:
            raise ValueError(f"Unknown phase: {phase}")

    def on_train_epoch_start(self):
        self._set_training_phase()

        phase = self.current_phase()
        phase_id = {"mean": 0.0, "variance": 1.0, "joint": 2.0}[phase]

        self.log("training_phase_id", phase_id, prog_bar=False, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch

        phase = self.current_phase()

        mean, log_var = self(x)

        mse = self.mse_loss(mean, y)
        nll = self.gaussian_nll_loss(mean, log_var, y)

        if phase == "mean":
            loss = mse

        elif phase == "variance":
            # Important: mean is fixed, variance learns to explain residuals
            nll_fixed_mean = self.gaussian_nll_loss(mean.detach(), log_var, y)
            loss = nll_fixed_mean
            nll = nll_fixed_mean

        elif phase == "joint":
            loss = nll

        else:
            raise ValueError(f"Unknown phase: {phase}")

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

        # val_loss is an alias for checkpointing convenience
        self.log("val_loss", val_nll, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_nll", val_nll, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mean_std", pred_std.mean(), prog_bar=True, on_step=False, on_epoch=True)

        return val_nll

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

    def configure_optimizers(self):
        """
        Use different regularization for mean and variance parts.

        shared + mean_head: mean_weight_decay
        log_var_head:      var_weight_decay
        """
        return torch.optim.AdamW(
            [
                {
                    "params": self.model.shared.parameters(),
                    "weight_decay": self.mean_weight_decay,
                },
                {
                    "params": self.model.mean_head.parameters(),
                    "weight_decay": self.mean_weight_decay,
                },
                {
                    "params": self.model.log_var_head.parameters(),
                    "weight_decay": self.var_weight_decay,
                },
            ],
            lr=self.lr,
        )