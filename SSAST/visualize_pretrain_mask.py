import math

import torch
import matplotlib.pyplot as plt

from ast_models import ASTModel
from pretrain_dataloader import DataloaderModule, show_np_spectrogram_file
from pretrain_ssast import calculate_total_patches, SSASTLightningModule


def visualize_ssast_reconstruction(num_samples=1):
    data_dir = [
            "/cluster/projects/uasc/Datasets/Data univ Bari/dataset",
            "/cluster/projects/uasc/Datasets/Dubrovnik/dataset",
            "/cluster/projects/uasc/Datasets/Fram Strait 2008-09/dataset",
            "/cluster/projects/uasc/Datasets/Blitvenica/dataset",
            "/cluster/projects/uasc/Datasets/Fram Strait 2017-18/dataset",
            "/cluster/projects/uasc/Datasets/Atwain 2017-18/dataset",
            "/cluster/projects/uasc/Datasets/Losinj/dataset",
        ]
    data_dir = ["C:\\Users\\ander\\Github\\Masters\\Pdata"]
    data_module = DataloaderModule(
        data_dirs=data_dir,
        train_val_test_split=(1, 0, 0),
        batch_size=1,
        num_workers=1,
        pin_memory=False,
    )
    data_module.setup()
    dataloader = data_module.train_dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    label_dim = 4
    fshape = 16
    tshape = 16
    fstride = 16
    tstride = 16
    input_fdim = 128
    input_tdim = 1024
    model_size = "base"
    checkpoint_path = r"/Results_from_cluster/First Pretraining/pretrained_best.pth.ckpt"
    #checkpoint_path = "C:\\Users\\ander\\Github\\Masters\\Results_from_cluster\\SSAST-Base-Patch-400.pth"
    mask_patch_size = math.floor(calculate_total_patches(input_tdim, input_fdim, tshape, fshape, tstride,
                                                           fstride) * 0.2)  # Patch mpc_mask 70% as suggested in SSAST paper
    """
    model = SSASTLightningModule.load_from_checkpoint(
        checkpoint_path,
        label_dim=2,
        fshape=fshape,
        tshape=tshape,
        fstride=fstride,
        tstride=tstride,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        model_size="base",
        learning_rate=1e-5,
        mask_patch_size=mask_patch_size,
        loss=torch.nn.CrossEntropyLoss()
    )

    """
    model = ASTModel(
        label_dim=4,  # Adjust based on your task
        fshape=16, tshape=16, fstride=16, tstride=16,
        input_fdim=128, input_tdim=1024,
        model_size='base',
        pretrain_stage=True,
    )
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda"))
    #model_weights = checkpoint["state_dict"]
    #checkpoint = {k.replace("model.", ""): v for k, v in model_weights.items()}
    model.load_state_dict(checkpoint, strict=True)
    #"""

    model.to(device)
    model.eval()
    for batch_idx, spectrograms in enumerate(dataloader):
        if num_samples <= batch_idx:
            break

        spectrograms = spectrograms.float().to(device)
        with torch.no_grad():
            mpc_pred, mpc_mask = model(spectrograms, mask_patch=mask_patch_size, task='visualize_mask', cluster=False)
            mpg_pred, mpg_mask = model(spectrograms, mask_patch=mask_patch_size, task='visualize_mask_mpg', cluster=False)

        mpc_pred = mpc_pred.cpu().numpy().squeeze().squeeze()
        mpc_mask = mpc_mask.cpu().numpy().squeeze().squeeze()
        mpg_pred = mpg_pred.cpu().numpy().squeeze().squeeze()
        mpg_mask = mpg_mask.cpu().numpy().squeeze().squeeze()
        spectrograms = spectrograms.cpu().numpy().squeeze().transpose(1,0)

        fig, axes = plt.subplots(3, 1, figsize=(30, 15))
        for ax, data, title in zip(axes,[spectrograms, mpc_pred, mpc_mask],["Input", "MPC Reconstructed (blue correct, red wrong)", "MPG Masked"]):
            ax.imshow(data,
                cmap='turbo',  # Good for bioacoustic visualization
                interpolation='none',
                origin='lower',
                aspect='auto',)
            ax.set_title(title)
        plt.savefig(f"results/mpc{batch_idx}.png", format='png')
        plt.show()

        fig, axes = plt.subplots(3, 1, figsize=(30, 15))
        for ax, data, title in zip(axes, [spectrograms, mpg_pred, mpg_mask],["Input", "MPG Reconstructed", "MPG Masked"]):
            ax.imshow(data,
                      cmap='turbo',  # Good for bioacoustic visualization
                      interpolation='none',
                      origin='lower',
                      aspect='auto', )
            ax.set_title(title)
        plt.savefig(f"results/mpg{batch_idx}.png", format='png')
        plt.show()

def main():
    visualize_ssast_reconstruction(5)
if __name__ == "__main__":
    main()