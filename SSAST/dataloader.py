from typing import Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchaudio.transforms import Spectrogram
from PIL import Image
import os


class Dataloader(Dataset):
    def __init__(self, spectrogram_dir: str, transform=None):
        super().__init__()
        self.spectrogram_dir = spectrogram_dir
        self.spectrogram_paths = [os.path.join(spectrogram_dir, file) for file in os.listdir(spectrogram_dir) if file.endswith('STENELLA.npy') or file.endswith('GRAMPO.npy')]
        #print(f"found {len(self.spectrogram_paths)} spectrograms")
        self.transform = transform

    def __len__(self):
        return len(self.spectrogram_paths)

    def __getitem__(self, idx):
        path = self.spectrogram_paths[idx]
        spectrogram = np.load(path)

        label_str = path.split('_')[-1].split('.')[0]
        if label_str == 'STENELLA':
            label = 1
        elif label_str == 'GRAMPO':
            label = 0
        elif label_str == 'QRY':
            label = -1
        else: label = -1

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        spectrogram = torch.permute(spectrogram, (1, 0))
        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram, label

class DataloaderModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "",
        train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Define image transformations
        self.image_transforms = transforms.Compose([
            #transforms.ToTensor(), # Do not use
            #transforms.Resize((128, 128)),
            #transforms.Normalize((0.5,), (0.5,))
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        full_dataset = Dataloader(
            spectrogram_dir=self.data_dir,
            transform=self.image_transforms
        )

        train_size = int(self.train_val_test_split[0] * len(full_dataset))
        val_size = int(self.train_val_test_split[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_set, val_set, test_set = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        self.data_train = train_set
        self.data_val = val_set
        self.data_test = test_set

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


def test_dataloader(spectrogram_dir: str, batch_size: int = 16):
    # Create FewShotDataset instance
    data_module = DataloaderModule(data_dir=output_spectrogram_folder, batch_size=1)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Load and print some batches
    for batch_idx, (spectrograms, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Spectrograms Shape: {spectrograms.shape}")  # Expecting [batch_size, 1, frequency_bins, time_frames]
        print(f"Labels: {labels}")

if __name__ == "__main__":
    output_spectrogram_folder = "dataset_positive_5sec_n"
    test_dataloader(output_spectrogram_folder, batch_size=16)
    """
    data_module = ImageToSpectrogramDataModule(data_dir=output_spectrogram_folder)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    for X, Y in train_loader:
        print(len(X))  # This will print the shape of the spectrograms in the batch
        break
    """
