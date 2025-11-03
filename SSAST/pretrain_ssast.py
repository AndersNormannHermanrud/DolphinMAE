import copy
import os
import random
import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, LearningRateMonitor, \
    DeviceStatsMonitor
from lightning.pytorch.utilities import grad_norm
from torch.nn import CrossEntropyLoss
from ast_models_debug import ASTModel
from pretrain_dataloader import DataloaderModule
import matplotlib.pyplot as plt
import numpy as np
import math

class NaNDetektorHook:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.has_nan = False

        for name, module in model.named_modules():
            hook = module.register_forward_hook(self.make_hook(name))
            self.hooks.append(hook)

    def make_hook(self, module_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                outputs = output
            else:
                outputs = (output,)

            for o in outputs:
                if o is not None and torch.is_tensor(o):
                    if torch.isnan(o).any() or torch.isinf(o).any():
                        print(f"\n🚨 NaN detected in module: {module_name}")

                        # Print input stats
                        print("Input stats:")
                        for idx, i in enumerate(input):
                            if i is not None and torch.is_tensor(i):
                                print(f" - Input[{idx}] shape: {i.shape} min: {i.min().item():.4e} max: {i.max().item():.4e} mean: {i.mean().item():.4e}")

                        # Print output stats
                        print("Output stats:")
                        print(f" - Output shape: {o.shape} min: {o.min().item():.4e} max: {o.max().item():.4e} mean: {o.mean().item():.4e}")

                        # Print parameters if the module has them
                        if hasattr(module, 'weight'):
                            print(f"Module weights stats: min {module.weight.data.min().item():.4e}, max {module.weight.data.max().item():.4e}")
                        if hasattr(module, 'bias') and module.bias is not None:
                            print(f"Module bias stats: min {module.bias.data.min().item():.4e}, max {module.bias.data.max().item():.4e}")

                        self.has_nan = True
        return hook

    def close(self):
        for hook in self.hooks:
            hook.remove()


class SSASTLightningModule(LightningModule):
    def __init__(self, label_dim, fshape, tshape, fstride, tstride, input_fdim, input_tdim, model_size, learning_rate,
                 mask_patch_size):
        super().__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters()

        self.model = ASTModel(
            label_dim=label_dim,
            fshape=fshape,
            tshape=tshape,
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            model_size=model_size,
            pretrain_stage=True,
            # load_pretrained_mdl_path="pretrained/SSAST-Base-Patch-400.pth"
        )
        self.learning_rate = learning_rate
        self.mask_patch_size = mask_patch_size
        self.nan_detector = None
        self.warmup_steps = 1000
        self.current_step = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=5e-7, betas=(0.95, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                                  verbose=True, cooldown=2)  # Same as SSAST
        # return [optimizer], [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",  # Monitored metric
                "interval": "epoch",
                "frequency": 1
            }
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.layer, norm_type=2)
        self.log_dict(norms)

    # def forward(self, x):
    #    acc, nce_loss = self.model(x, task='pretrain_mpc', cluster=False, mask_patch=self.mask_patch_size)
    #    mse_loss = self.model(x, task='pretrain_mpg', cluster=False, mask_patch=self.mask_patch_size)
    #    loss = nce_loss + 10 * mse_loss
    #    return loss

    def training_step(self, batch, batch_idx):
        batch = batch.float().to(self.device)
        acc, nce_loss = self.model(batch, task='pretrain_mpc', cluster=True, mask_patch=self.mask_patch_size)
        acc, nce_loss = acc.mean(), nce_loss.mean()
        mse_loss = self.model(batch, task='pretrain_mpg', cluster=True, mask_patch=self.mask_patch_size)
        mse_loss = mse_loss.mean()
        loss = nce_loss + 10 * mse_loss

        # Warmup
        optimizer = self.optimizers()
        if self.current_step < self.warmup_steps:
            warmup_lr = self.learning_rate * float(self.current_step + 1) / float(self.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        if self.current_step == 100:
            if self.nan_detector is None:
                self.nan_detector = NaNDetektorHook(self.model)
        self.current_step += 1

        self.log('nce_loss', nce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("learning rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)
        # Plot Variance inn loss

        return loss
        # investigative approach to comparison
        # Improve dolphins model, then show comparison to other model
        # How do pretraining with model from this domain help? Use good metrics and logging to refer to. Also, reconstruction performance and comparison (Raw and XAI). Maybe Try on other underwater domain.
        # Become more serius on notes, find Research question! Draft research question. Write down the training process
        # Possible research question, see how different hyperparameter optimization works, use domain knowledge for this
        # Possible research question, how to adapt a transformer model for the underwater domain. FUTURE WORK
        # Revisit data clustering? Probably not. Overlap slizing with clustering, to balance the different kinds of noise
        # Worked on it long, what can be improved, what would make sense to change, remove or add.

        # To next meeting. Have draft research questions and documentation about current training pipeline ready

        #Check layernorm
        #Noise, find edges/sharp differences. MEL melds them together
        #Loss mean, change this? Cross entropy?
        #Input spectrogram shape
        #Initial loss/acc slightly better for dolph compared to other. Suspect low sample rate throws off

        # Can i use DOSIT site as a source?

    def validation_step(self, batch, batch_idx):
        batch = batch.float().to(self.device)
        acc, nce_loss = self.model(batch, task='pretrain_mpc', cluster=True, mask_patch=self.mask_patch_size)
        acc, nce_loss = acc.mean(), nce_loss.mean()
        mse_loss = self.model(batch, task='pretrain_mpg', cluster=True, mask_patch=self.mask_patch_size)
        mse_loss = mse_loss.mean()
        loss = nce_loss + 10 * mse_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss



def calculate_total_patches(tdim, fdim, tshape, fshape, tstride, fstride):
    """
    Calculate the total number of patches based on the input spectrogram size, patch shape, and strides.
    """
    time_patches = (tdim - tshape) // tstride + 1
    freq_patches = (fdim - fshape) // fstride + 1
    total_patches = time_patches * freq_patches

    return total_patches


def main():
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    print(f"is using cuda? {torch.cuda.is_available()}")

    # TODO reduce to doplphin datasets
    # TODO log better metrics (loss/acc across epoch pr epoch, merge logs
    data_dirs = [
        "/cluster/projects/uasc/Datasets/Data univ Bari/dataset",
        "/cluster/projects/uasc/Datasets/Dubrovnik/dataset",
        # "/cluster/projects/uasc/Datasets/Fram Strait 2008-09/dataset",
        # "/cluster/projects/uasc/Datasets/Blitvenica/dataset",
        # "/cluster/projects/uasc/Datasets/Fram Strait 2017-18/dataset",
        # "/cluster/projects/uasc/Datasets/Atwain 2017-18/dataset",
        "/cluster/projects/uasc/Datasets/Losinj/dataset",
    ]
    batch_size = 50
    train_val_test_split = (0.8, 0.2, 0)

    # Manual computer testing lines
    #data_dirs = ["C:\\Users\\ander\\Github\\Masters\\Pdata"]

    data_module = DataloaderModule(
        data_dirs=data_dirs,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        num_workers=18,
        pin_memory=True
    )
    # data_module.setup()

    label_dim = 4
    fshape = 16
    tshape = 16
    fstride = 16
    tstride = 16
    input_fdim = 128
    input_tdim = 1024
    model_size = "base"

    #TODO check if masking all useful information in a spectrogram can lead to nah values, maybe reduce
    masked_patch_size = math.floor(calculate_total_patches(input_tdim, input_fdim, tshape, fshape, tstride,
                                                           fstride) * 0.7)  # Patch mask 70% as suggested in SSAST paper
    print(f"Working with a patch size of {masked_patch_size} (70% of total patches)")

    model = SSASTLightningModule(
        label_dim=label_dim,
        fshape=fshape,
        tshape=tshape,
        fstride=fstride,
        tstride=tstride,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        model_size=model_size,
        learning_rate=5e-4,
        mask_patch_size=masked_patch_size,
    )

    # Callbacks, additional features while training
    logger = TensorBoardLogger("lightning_logs", name="ssast_pretraining")
    #callback = MetricsCallback()
    model_summary = ModelSummary(max_depth=12)
    device_monitor = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Checkpoint for best weights (only weights)
    checkpoint_callback_weights = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        dirpath="results/checkpoints/",
        filename="pretrained_best_weights",
        save_weights_only=True,
    )

    # Checkpoint for best full model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        dirpath="results/checkpoints/",
        filename="pretrained_best",
        save_weights_only=False,
        every_n_epochs=2,
    )

    # Checkpoint for latest epoch
    epoch_checkpoint = ModelCheckpoint(
        dirpath="results/checkpoints/",
        filename="pretraining_latest_epoch",
        save_top_k=-1,
        every_n_epochs=1,
        save_weights_only=False,
    )

    # Training setup
    trainer = Trainer(
        detect_anomaly=True,
        max_epochs=1000,
        devices=1,
        num_nodes=1,
        accelerator='gpu',
        strategy='auto',#strategy='ddp_find_unused_parameters_true',
        precision='16-mixed',
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=100,
        callbacks=[device_monitor, lr_monitor, checkpoint_callback, model_summary,
                   checkpoint_callback_weights, epoch_checkpoint],
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
        # ckpt_path = "results/checkpoints/pretraining_latest_epoch.ckpt",
    )


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("results/pretrain_loss.png", format='png')
    # plt.show()


"""
class MetricsCallback(Callback):
    #PyTorch Lightning metric callback.
    #Just saves the result a little better than the lightning logs

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics.append(copy.deepcopy(trainer.callback_metrics))
        train_losses = [metrics['train_loss_epoch'].item() for metrics in self.metrics if
                        'train_loss_epoch' in metrics]
        val_losses = [metrics['val_loss'].item() for metrics in self.metrics if 'val_loss' in metrics]
        plot_loss(train_losses, val_losses)

    def on_train_end(self, trainer, pl_module):
        torch.save(self.metrics, os.path.join("../results", "pretrain_metrics.pt"))
        train_losses = [metrics['train_loss_epoch'].item() for metrics in self.metrics if
                        'train_loss_epoch' in metrics]
        val_losses = [metrics['val_loss'].item() for metrics in self.metrics if 'val_loss' in metrics]
        plot_loss(train_losses, val_losses)
"""

if __name__ == "__main__":
    main()
