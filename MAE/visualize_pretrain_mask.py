import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models_mae import MaskedAutoencoderViT  # update with your correct import path
from pretrain_audiomae import MaeLightningModule
from pretrain_dataloader import DataloaderModule

@torch.no_grad()
def visualize_audiomae_reconstruction(checkpoint_path, num_samples=1, mask_ratio=0.75):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaeLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.to(device).eval()
    model = model.model

    data_module = DataloaderModule(
        data_dirs=["C:\\Users\\ander\\Github\\Masters\\Pdata\\Thesis Examples"],
        train_val_test_split=(1, 0, 0),
        batch_size=1, num_workers=1, pin_memory=False
    )
    data_module.setup()
    dataloader = data_module.train_dataloader()

    for idx, batch in enumerate(dataloader):
        #if idx >= num_samples:
            #break

        x = batch.to(device).float()
        with torch.no_grad():
            x_encoded, mask, ids_restore, _ = model.forward_encoder(x, mask_ratio=mask_ratio)
            x_recon, _, _ = model.forward_decoder(x_encoded, ids_restore)
        x_orig = model.patchify(x).cpu().squeeze(0)  # (L, D)
        x_recon = x_recon.cpu().squeeze(0)
        mask = mask.cpu().squeeze(0)  # (L,)

        print(f"x_orig: {x_orig.shape}, x_recon: {x_recon.shape}")

        # Keep only masked parts
        recon_only_masked = x_orig.clone()
        recon_only_masked[mask == 0] = float("nan")  # Keep unmasked as NaN for clarity
        recon_only_masked[mask == 1] = x_recon[mask == 1]

        # Unpatchify
        original = model.unpatchify(x_orig.unsqueeze(0)).squeeze().cpu().numpy()
        reconstructed = model.unpatchify(x_recon.unsqueeze(0)).squeeze().cpu().numpy()
        masked_recon = model.unpatchify(recon_only_masked.unsqueeze(0)).squeeze().cpu().numpy()
        mask_image = mask.reshape(256 // 16 , 8).float().numpy()  # assumes 16x16 patches, (1024/16)x(128/16) = 64x8

        # Print statistics
        orig_masked_patches = x_orig[mask == 1]
        recon_masked_patches = x_recon[mask == 1]

        mean_orig = orig_masked_patches.mean().item()
        mean_recon = recon_masked_patches.mean().item()
        std_orig = orig_masked_patches.std().item()
        std_recon = recon_masked_patches.std().item()

        print(f"[{idx}] Original (masked) mean: {mean_orig:.3f}, std: {std_orig:.3f}")
        print(f"[{idx}] Reconstructed mean:      {mean_recon:.3f}, std: {std_recon:.3f}")

        # === Compute shared color scale ===
        vmin = np.min(original)
        vmax = np.max(original)

        fig, axes = plt.subplots(2, 1, figsize=(20, 16))

        # --- Row 1: Original ---
        axes[0].imshow(original.T, cmap="turbo", interpolation="none", origin="lower", aspect="auto", vmin=vmin,
                       vmax=vmax)
        axes[0].set_title("Original Spectrogram", fontsize=24)
        axes[0].axis("off")

        # --- Row 2: Reconstructed scaled after original---
        axes[1].imshow(masked_recon.T, cmap="turbo", interpolation="none", origin="lower", aspect="auto", vmin=vmin,
                       vmax=vmax)
        axes[1].set_title("Reconstructed Spectrogram", fontsize=24)
        axes[1].axis("off")

        # --- Row 3: Reconstructed Only (Independent color scale) ---
        #axes[2].imshow(masked_recon.T, cmap="turbo", interpolation="none", origin="lower", aspect="auto")
        #axes[2].set_title("Reconstructed Only")
        #axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"results/audiomae_recon_{idx}_combined.png", dpi=200)
        plt.show()
        plt.close()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    visualize_audiomae_reconstruction(
        checkpoint_path=r"C:\Users\ander\Github\Masters\Results_from_cluster\mae\Final_Experiment\Epoch 41\pretrained_best-v1.ckpt",  # update this path
        num_samples=5,
        mask_ratio=0.70
    )