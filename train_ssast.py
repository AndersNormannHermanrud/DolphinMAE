import copy
import os
import random
import time

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from ast_models import ASTModel
from dataloader import DataloaderModule, Dataloader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SSASTFewShotLightningModule(LightningModule):
    def __init__(self, label_dim, fshape, tshape, fstride, tstride, input_fdim, input_tdim, model_size, learning_rate=1e-4):
        super().__init__()
        self.model = ASTModel(
            label_dim=label_dim,
            fshape=fshape,
            tshape=tshape,
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            model_size=model_size,
            pretrain_stage=False,
            load_pretrained_mdl_path="results/pretrained.pth",
            #load_pretrained_mdl_path="pretrained/SSAST-Base-Patch-400.pth",
        )
        self.Loss = CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x, task='ft_avgtok')

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float().to(self.device)  # Ensure input type matches model requirements and move to GPU if available
        y = y.to(self.device)  # Move labels to the same device as the model
        embeddings = self.model(x, task='ft_avgtok')  # Extract embeddings
        loss = self.Loss(embeddings, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float().to(self.device)  # Ensure input type matches model requirements and move to GPU if available
        y = y.to(self.device)  # Move labels to the same device as the model
        embeddings = self.model(x, task='ft_avgtok')  # Extract embeddings
        val_loss = self.Loss(embeddings, y)
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        """
            # diff lr optimizer
            mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
            mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
            base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
            mlp_params = [i[1] for i in mlp_params]
            base_params = [i[1] for i in base_params]
            # only finetuning small/tiny models on balanced audioset uses different learning rate for mlp head
            print('The mlp header uses {:d} x larger lr'.format(args.head_lr))
            optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
            mlp_lr = optimizer.param_groups[1]['lr']
            lr_list = [args.lr, mlp_lr]
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_end(self):
        saved_model = torch.nn.DataParallel(self.model)
        torch.save(saved_model.state_dict(), 'results/finetuned.pth')

    @torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    def get_logits(self, dataloader):
        self.eval()
        i = 0
        logits, labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                i += 1
                x, y = batch
                y_hat = self(x)
                logits.extend(y_hat.cpu().numpy())
                labels.extend(y.cpu().numpy())
                if i % 50 == 0:
                    print(i)
                if i > 600:
                    break
        return logits, labels

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
        torch.save(self.metrics, os.path.join("results", "finetune_metrics.pt"))

def main():
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    data_dir = "/cluster/projects/uasc/anders/datasets/dataset_trunct"
    query_data_dir = "/cluster/projects/uasc/anders/datasets/dataset_query_trunct"
    #dataloader_query = DataLoader(ImageToSpectrogramDataset(query_data_dir), batch_size=256, shuffle=False, num_workers=4, pin_memory=False)
    callback = MetricsCallback()
    batch_size = 100
    train_val_test_split = (0.8, 0.2, 0)
    data_module = DataloaderModule(
        data_dir=data_dir,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        num_workers=28,
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

    model = SSASTFewShotLightningModule(
        label_dim=label_dim,
        fshape=fshape,
        tshape=tshape,
        fstride=fstride,
        tstride=tstride,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        model_size=model_size,
        learning_rate=1e-6
    )
    


    time_start = time.time()
    # Training setup, need to attach the mixed precision float
    trainer = Trainer(
        max_epochs=60,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision='16-mixed' if torch.cuda.is_available() else 32,  # Use mixed precision if possible
        callbacks=[callback, checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, data_module)
    time_taken = time.time() - time_start
    print(f"Time taken to train: {time_taken}")
    train_losses = [metrics['train_loss_epoch'].item() for metrics in callback.metrics if 'train_loss_epoch' in metrics]
    val_losses = [metrics['val_loss'].item() for metrics in callback.metrics if 'val_loss' in metrics]
    plot_loss(train_losses, val_losses)

    # Logits before training
    logits, labels = model.get_logits(data_module.train_dataloader())
    #logits_q, labels_q = model.get_logits(dataloader_query)
    #logits = np.concatenate((logits, logits_q), axis=0)
    #labels = np.concatenate((labels, labels_q), axis=0)
    plot_logits(logits, labels, method="tsne", name="Before Training Query Set dim2")



def plot_logits(logits, labels, method='tsne', n_components=2, name="raw logits"):
    if logits.shape[1] > 2:
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        else:  # method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        # else:
        # raise ValueError("Unsupported dimensionality reduction method. Use 'pca' or 'tsne'.")
        reduced_logits = reducer.fit_transform(logits)
    else:
        reduced_logits = logits

    logits_class_0 = reduced_logits[labels == 0]
    logits_class_1 = reduced_logits[labels == 1]
    logits_class_2 = reduced_logits[labels == -1]
    #logits_class_3 = reduced_logits[labels == -99]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=logits_class_2[:, 0], y=logits_class_2[:, 1], color='green', label='Query')
    sns.scatterplot(x=logits_class_0[:, 0], y=logits_class_0[:, 1], color='blue', label='Negative')
    sns.scatterplot(x=logits_class_1[:, 0], y=logits_class_1[:, 1], color='red', label='Whale')
    #sns.scatterplot(x=logits_class_3[:, 0], y=logits_class_1[:, 1], color='black', label='Other')

    plt.title(name)
    plt.legend()
    plt.savefig(name + ".png", format='png')
    #plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=logits_class_0[:, 0], y=logits_class_0[:, 1], color='blue', label='Negative')
    sns.scatterplot(x=logits_class_1[:, 0], y=logits_class_1[:, 1], color='red', label='Whale')
    # sns.scatterplot(x=logits_class_2[:, 0], y=logits_class_2[:, 1], color='green', label='Query')
    plt.title(name)
    plt.legend()
    plt.savefig(name + " only traindata" + ".png", format='png')
    #plt.show()


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("results/finetune_loss.png", format='png')
    #plt.show()


if __name__ == "__main__":
    main()
