import copy
import os
import random
import json
import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping, ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from ast_models import ASTModel
from pretrain_dataloader import DataloaderModule
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


class SSASTLightningModule(LightningModule):
    def __init__(self, label_dim, fshape, tshape, fstride, tstride, input_fdim, input_tdim, model_size, learning_rate, mask_patch_size, loss):
        super().__init__()
        #self.automatic_optimization = False

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
            #load_pretrained_mdl_path="pretrained/SSAST-Base-Patch-400.pth"
        )
        self.Loss = loss #CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.mask_patch_size = mask_patch_size

    #def forward(self, x):
    #    acc, nce_loss = self.model(x, task='pretrain_mpc', cluster=False, mask_patch=self.mask_patch_size)
    #    mse_loss = self.model(x, task='pretrain_mpg', cluster=False, mask_patch=self.mask_patch_size)
    #    loss = nce_loss + 10 * mse_loss
    #    return loss

    def training_step(self, batch, batch_idx):
        x= batch
        x = x.float().to(self.device)
        acc, nce_loss = self.model(x, task='pretrain_mpc', cluster=False, mask_patch=self.mask_patch_size)
        mse_loss = self.model(x, task='pretrain_mpg', cluster=False, mask_patch=self.mask_patch_size)
        loss = nce_loss + 10 * mse_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log LR
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning rate", lr, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.float().to(self.device)
        acc, nce_loss = self.model(x, task='pretrain_mpc', cluster=False, mask_patch=self.mask_patch_size)
        mse_loss = self.model(x, task='pretrain_mpg', cluster=False, mask_patch=self.mask_patch_size)
        loss = nce_loss + 10 * mse_loss
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=5e-7, betas=(0.95, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=4, verbose=True) # Same as SSAST
        #return [optimizer], [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitored metric
                "interval": "epoch",
                "frequency": 1
            }
        }

    def on_train_end(self):
        saved_model = torch.nn.DataParallel(self.model)
        torch.save(saved_model.state_dict(), 'results/pretrained.pth')


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback.
    Just saves the result a little better than the lightning logs
    """
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics.append(copy.deepcopy(trainer.callback_metrics))

    def on_train_end(self, trainer, pl_module):
        #print(self.metrics)
        torch.save(self.metrics, os.path.join("results", "pretrain_metrics.pt"))

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
    data_dirs = [
        "/cluster/projects/uasc/Datasets/Data univ Bari/dataset",
        "/cluster/projects/uasc/Datasets/Dubrovnik/dataset",
        "/cluster/projects/uasc/Datasets/Fram Strait 2008-09/dataset",
        "/cluster/projects/uasc/Datasets/Blitvenica/dataset",
        "/cluster/projects/uasc/Datasets/Fram Strait 2017-18/dataset",
        "/cluster/projects/uasc/Datasets/Atwain 2017-18/dataset",
        "/cluster/projects/uasc/Datasets/Losinj/dataset",
    ]
    batch_size = 80
    train_val_test_split = (0.8, 0.2, 0)

    data_module = DataloaderModule(
        data_dir=data_dirs,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True
    )
    data_module.setup()

    label_dim = 2
    fshape = 16
    tshape = 16
    fstride = 16
    tstride = 16
    input_fdim = 128
    input_tdim = 1000
    model_size = "base"
    masked_patch_size = math.floor(calculate_total_patches(input_tdim, input_fdim, tshape, fshape, tstride, fstride)*0.7) # Patch mask 70% as suggested in SSAST paper
    print(f"Working with a patch size of {masked_patch_size} (70% of total patches)")

    loss = CrossEntropyLoss()
    model = SSASTLightningModule(
        label_dim=label_dim,
        fshape=fshape,
        tshape=tshape,
        fstride=fstride,
        tstride=tstride,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        model_size=model_size,
        learning_rate=1e-3,
        mask_patch_size = masked_patch_size,
        loss=loss,
    )

    # Callbacks, additional features while training
    logger = TensorBoardLogger("lightning_logs", name="ssast_pretraining")
    callback = MetricsCallback()
    model_summary = ModelSummary(max_depth=12)
    device_monitor = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min", # Go for minimizing loss since accuracy does not exist
        dirpath="results/checkpoints/",
        filename="pretrained_best.pth",
    )

    # Training setup
    trainer = Trainer(
        max_epochs=40,
        devices=1,
        num_nodes=1,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        precision='16-mixed',
        logger=logger,
        callbacks=[callback,device_monitor,lr_monitor, checkpoint_callback, model_summary],
    )
    print("Starting training")
    trainer.fit(model, data_module)
    train_losses = [metrics['train_loss_epoch'].item() for metrics in callback.metrics if 'train_loss_epoch' in metrics]
    val_losses = [metrics['val_loss'].item() for metrics in callback.metrics if 'val_loss' in metrics]
    plot_loss(train_losses, val_losses)


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("results/pretrain_loss.png", format='png')
    #plt.show()


if __name__ == "__main__":
    main()
