import os
import sys

#print(f"Running Path: {os.getcwd()}")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, LearningRateMonitor, \
    DeviceStatsMonitor
from torch.nn import CrossEntropyLoss
from pretrain_dataloader import DataloaderModule
#import models_mae_debug as models_mae
import models_mae
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/util")


class MaeLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters()

        model = 'mae_vit_base_patch16'
        self.mask_ratio = 0.85

        self.model = models_mae.__dict__[model](norm_pix_loss=False,
                                                in_chans=1, audio_exp=True,
                                                img_size=(1024, 128),
                                                alpha=0.0, mode=0, use_custom_patch=False,
                                                split_pos=False, pos_trainable=False, use_nce=False,
                                                decoder_mode=0,
                                                mask_2d=False, mask_t_prob=0.7, mask_f_prob=0.3,
                                                no_shift=False,
                                                )
        self.learning_rate = 1e-4
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
                "monitor": "MSE_loss_epoch",  # Monitored metric
                "interval": "epoch",
                "frequency": 1
            }
        }

    # def forward(self, x):
    #    acc, nce_loss = self.model(x, task='pretrain_mpc', cluster=False, mask_patch=self.mask_patch_size)
    #    mse_loss = self.model(x, task='pretrain_mpg', cluster=False, mask_patch=self.mask_patch_size)
    #    loss = nce_loss + 10 * mse_loss
    #    return loss



    def training_step(self, batch, batch_idx):
        # x = batch
        #print(f"Batch.shape {batch.shape}")
        #for spectrogram in batch:
        #    show_np_spectrogram_file(spectrogram[0,:,:].cpu().numpy())
        with torch.cuda.amp.autocast():
            loss_a, _, _, _ = self.model.forward(batch, mask_ratio=self.mask_ratio)
        loss_value = loss_a.item()
        loss_total = loss_a
        if not torch.isfinite(loss_a).all():
            print(loss_a)
            raise ValueError("Loss is infinite")

        # Warmup
        optimizer = self.optimizers()
        if self.current_step < self.warmup_steps:
            warmup_lr = self.learning_rate * float(self.current_step + 1) / float(self.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        self.current_step += 1

        self.log('MSE_loss', loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("learning rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        # Plot Variance inn loss
        return loss_total
        # investigative approach to comparison
        # Improve dolphins model, then show comparison to other model
        # How do pretraining with model from this domain help? Use good metrics and logging to refer to. Also, reconstruction performance and comparison (Raw and XAI). Maybe Try on other underwater domain.
        # Become more serius on notes, find Research question! Draft research question. Write down the training process
        # Possible research question, see how different hyperparameter optimization works, use domain knowledge for this
        # Possible research question, how to adapt a transformer model for the underwater domain. FUTURE WORK
        # Revisit data clustering? Probably not. Overlap slizing with clustering, to balance the different kinds of noise
        # Worked on it long, what can be improved, what would make sense to change, remove or add.

        # To next meeting. Have draft research questions and documentation about current training pipeline ready

        # Check layernorm
        # Noise, find edges/sharp differences. MEL melds them together
        # Loss mean, change this? Cross entropy?
        # Input spectrogram shape
        # Initial loss/acc slightly better for dolph compared to other. Suspect low sample rate throws off

    def validation_step(self, batch, batch_idx):
        with torch.cuda.amp.autocast():
            loss_a, _, _, _ = self.model(batch, mask_ratio=self.mask_ratio)
        loss_value = loss_a.item()
        loss_total = loss_a
        self.log('val_MSE_loss', loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss_total

def show_np_spectrogram_file(waveform, file_name=""):
    plt.figure(figsize=(25, 5))
    title = f"waveform min: {waveform.min()}, waveform max: {waveform.max()}"
    img = plt.imshow(
        waveform.T,
        cmap='turbo',  # Good for bioacoustic visualization
        interpolation='none',
        origin='lower',
        aspect='auto',
        # vmin=vmin, vmax=vmax  # Set dynamic range
    )
    plt.ylim(0, waveform.T.shape[0])  # Frequency bins
    plt.xlim(0, waveform.T.shape[1])  # Time frames
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.colorbar(img, label="dB")
    plt.title(title)
    plt.show()

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
    #torch.autograd.set_detect_anomaly(True)

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

    # Manual computer testing lines
    #data_dirs = ["C:\\Users\\ander\\Github\\Masters\\Pdata"]

    #print("You have local testing settings on, batch size wrong")
    batch_size = 40
    train_val_test_split = (0.8, 0.2, 0)

    data_module = DataloaderModule(
        data_dirs=data_dirs,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        num_workers=18,
        pin_memory=True
    )
    # data_module.setup()
    # masked_patch_size = math.floor(calculate_total_patches(input_tdim, input_fdim, tshape, fshape, tstride,fstride) * 0.7)  # Patch mask 70% as suggested in SSAST paper
    # print(f"Working with a patch size of {masked_patch_size} (70% of total patches)")


    model = MaeLightningModule()

    # Callbacks, additional features while training
    logger = TensorBoardLogger("lightning_logs", name="audiomae_pretraining")
    model_summary = ModelSummary(max_depth=12)
    device_monitor = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Checkpoint for best weights (only weights)
    checkpoint_callback_weights = ModelCheckpoint(
        monitor="val_MSE_loss",
        save_top_k=1,
        mode="min",
        dirpath="results/MAEcheckpoints/",
        filename="pretrained_best_weights",
        save_weights_only=True,
    )

    # Checkpoint for best full model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_MSE_loss",
        save_top_k=1,
        mode="min",
        dirpath="results/MAEcheckpoints/",
        filename="pretrained_best",
        save_weights_only=False,
        every_n_epochs=2,
    )

    # Checkpoint for latest epoch
    epoch_checkpoint = ModelCheckpoint(
        dirpath="results/MAEcheckpoints/",
        filename="pretraining_latest_epoch",
        save_top_k=-1,
        every_n_epochs=1,
        save_weights_only=False,
    )

    # Training setup
    trainer = Trainer(
        max_epochs=200,
        devices=1,
        num_nodes=1,
        accelerator='gpu',
        strategy='auto',
        precision='16-mixed',
        logger=logger,
        log_every_n_steps=100,
        callbacks=[device_monitor, lr_monitor, checkpoint_callback, model_summary,
                   checkpoint_callback_weights, epoch_checkpoint],
    )

    path = "/cluster/projects/uasc/anders/ssast_pretraining/results/MAEcheckpoints/pretraining_latest_epoch.ckpt"
    #ckpt_path = [i for i in os.listdir(path) if "latest" in i][0]

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path = path #+ "/" + ckpt_path,
    )


if __name__ == "__main__":
    main()
