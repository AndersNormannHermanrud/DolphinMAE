import os
import sys

from torch import nn

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
from finetune_dataloader import DataloaderModule
#import models_mae_debug as models_mae
import models_mae
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

import argparse, pathlib
from torch.serialization import add_safe_globals

# Allow-list benign globals used by many training scripts
add_safe_globals([argparse.Namespace, pathlib.PosixPath, pathlib.PurePosixPath])

def _choose_state_dict(root: dict | OrderedDict):
    # Try common containers in order
    for key in ("state_dict", "model", "model_state", "net", "weights", "params"):
        v = root.get(key)
        if isinstance(v, (dict, OrderedDict)) and any(torch.is_tensor(x) for x in v.values()):
            return v
    # Flat state_dict at root?
    if isinstance(root, (dict, OrderedDict)) and any(torch.is_tensor(v) for v in root.values()):
        return root
    # Flattened keys like 'model.xxx'
    if isinstance(root, (dict, OrderedDict)) and any(isinstance(k, str) and k.startswith("model.") for k in root.keys()):
        return {k[len("model."):]: v for k, v in root.items() if torch.is_tensor(v)}
    return None

def load_backbone_weights(pretrained_path: str, model: nn.Module, *, trusted: bool = True) -> None:
    # 1) Safe path first: extract tensors only
    try:
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[load_backbone_weights] weights_only=True failed: {e}")
        if not trusted:
            raise
        # 2) Trusted fallback: full unpickle (can execute code if file is malicious)
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    # 3) Resolve to a real state_dict
    if isinstance(ckpt, nn.Module):
        src_sd = ckpt.state_dict()
    elif isinstance(ckpt, (dict, OrderedDict)):
        src_sd = _choose_state_dict(ckpt)
        if src_sd is None:
            raise ValueError("Could not locate a tensor state_dict in the checkpoint.")
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    # 4) Clean keys and drop non-encoder params
    cleaned = {}
    for k, v in src_sd.items():
        if not torch.is_tensor(v):
            continue
        kk = k
        for pref in ("model.", "module.", "encoder.", "backbone."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        if kk.startswith("decoder") or kk.startswith("head"):
            continue
        cleaned[kk] = v

    # 5) Skip shape-mismatched tensors (e.g., pos_embed if grid differs)
    msd = model.state_dict()
    pop_keys = [k for k, v in cleaned.items() if k in msd and msd[k].shape != v.shape]
    for k in pop_keys:
        print(f"[warn] shape mismatch {k}: ckpt {tuple(cleaned[k].shape)} vs model {tuple(msd[k].shape)} -> skipping")
        cleaned.pop(k)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[load_backbone_weights] Missing={len(missing)} Unexpected={len(unexpected)}")
    if missing:    print("  missing (first 10):", missing[:10])
    if unexpected: print("  unexpected (first 10):", unexpected[:10])


class MaeLightningModule(LightningModule):
    def __init__(self,pretrained_path, num_classes):
        super().__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters()

        model = 'mae_vit_base_patch16'
        self.mask_ratio = 0.70
        self.num_classes = num_classes
        self.mask_t_prob = 0.3
        self.mask_f_prob =0.3

        self.model = models_mae.__dict__[model](
            norm_pix_loss=False, in_chans=1, audio_exp=True,
            img_size=(128,1024), alpha=0.0, mode=0, use_custom_patch=False,
            split_pos=False, pos_trainable=False, use_nce=False, decoder_mode=0,
            mask_2d=True, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob, no_shift=False
        )

        #state = torch.load(pretrained_path, map_location='cpu')
        #sd = state.get('state_dict', state)
        load_backbone_weights(pretrained_path, self.model)

        # Strip possible lightning prefixes and skip decoder / classification head
        #new_sd = {}
        #for k, v in sd.items():
         #   kk = k
         #   kk = kk.replace('model.', '') if kk.startswith('model.') else kk
         #   if 'decoder' in kk or 'head' in kk:
        #        continue
        #    new_sd[kk] = v
        #missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
        #print('Loaded pretrain. Missing:', missing, ' Unexpected:', unexpected)


        self.criterion = nn.BCEWithLogitsLoss()
        self.head = nn.Linear(self.model.embed_dim, self.num_classes)

        self.learning_rate = 2e-5

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.learning_rate * 1e-2)
        # return [optimizer], [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                #"monitor": "loss",  # Monitored metric
                "interval": "epoch",
            }
        }

    def forward(self, x, train_mask=True):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if self.training and train_mask:
            toks, _, _, _ = self.model.forward_encoder(x, mask_ratio=0.0, mask_2d=True)  # (B, 1+N, C)
        else:
            toks = self.model.forward_encoder_no_mask(x)  # (B, 1+N, C)

        # average pool **patch tokens** (ignore CLS at index 0)
        feats = toks[:, 1:, :].mean(dim=1)  # (B, C)
        return self.head(feats)


    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
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

        # Check layernorm
        # Noise, find edges/sharp differences. MEL melds them together
        # Loss mean, change this? Cross entropy?
        # Input spectrogram shape
        # Initial loss/acc slightly better for dolph compared to other. Suspect low sample rate throws off

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x, train_mask=False)
        loss = self.criterion(outputs, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

def main():
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    print(f"is using cuda? {torch.cuda.is_available()}")
    #torch.autograd.set_detect_anomaly(True)

    # TODO reduce to doplphin datasets
    # TODO log better metrics (loss/acc across epoch pr epoch, merge logs
    location_filter = [
        "Taranto",
        "Cmhs",
        "Dubrovnik",
        #"/cluster/projects/uasc/Datasets/Fram Strait 2008-09/dataset",
        #"/cluster/projects/uasc/Datasets/Blitvenica/dataset",
        #"/cluster/projects/uasc/Datasets/Fram Strait 2017-18/dataset",
        #"/cluster/projects/uasc/Datasets/Atwain 2017-18/dataset",
        "Losinj",
    ]
    label_file = "/cluster/projects/uasc/Datasets/labels.xlsx"
    pretrained_path = "/cluster/projects/uasc/anders/ssast_pretraining/MAE/pretrained.pth"

    batch_size = 64
    train_val_test_split = (0.8, 0.2, 0)


    # Manual testing lines
    #label_file = "C:\\Users\\ander\\Github\\Masters\\Pdata\\labels_local_test.xlsx"
    #batch_size = 1
    #pretrained_path = r"C:\Users\ander\Github\Masters\MAE\pretrained.pth"


    data_module = DataloaderModule(
        label_file_path=label_file,
        location_filter=location_filter,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True
    )

    num_species = data_module.num_classes
    print(f"Setting up model with {num_species} classes")

    model = MaeLightningModule(pretrained_path=pretrained_path, num_classes=num_species)

    # Callbacks, additional features while training
    logger = TensorBoardLogger("lightning_logs", name="audiomae_finetuning", version="base")
    model_summary = ModelSummary(max_depth=12)
    device_monitor = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Checkpoint for best weights (only weights)
    checkpoint_callback_weights = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath="results/MAEcheckpoints/",
        filename="base_finetune_best_weights",
        save_weights_only=True,
    )

    # Checkpoint for best full model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath="results/MAEcheckpoints/",
        filename="base_finetune_best",
        save_weights_only=False,
        every_n_epochs=2,
    )

    # Checkpoint for latest epoch
    epoch_checkpoint = ModelCheckpoint(
        dirpath="results/MAEcheckpoints/",
        filename="base_finetune_latest_epoch",
        save_top_k=-1,
        every_n_epochs=1,
        save_weights_only=False,
    )

    # Training setup
    trainer = Trainer(
        max_epochs=100,
        devices=1,
        num_nodes=1,
        accelerator='gpu',
        strategy='auto', #strategy='ddp',
        precision='16-mixed',
        logger=logger,
        log_every_n_steps=50,
        callbacks=[lr_monitor, checkpoint_callback, model_summary,
                   checkpoint_callback_weights, epoch_checkpoint],
        profiler="simple"
    )

    trainer.fit(
        model=model,
        datamodule=data_module
    )


if __name__ == "__main__":
    main()
