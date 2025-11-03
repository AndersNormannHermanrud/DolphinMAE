import math
import pathlib
import random
import re
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, IterableDataset
# from util import show_np_spectrogram_file
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import os

def convert_numpy_to_fbank(waveform, sample_rate, max_sample_rate=120000):
    """
    Converts a numpy array (audio segment) to a mel spectrogram using torchaudio's fbank function.
    """
    if sample_rate > max_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=max_sample_rate)
        sample_rate = max_sample_rate
        waveform = resampler(waveform)

    waveform = waveform - waveform.mean()
    frame_length = 25
    frame_shift = 10

    #win_len_frames = int(round(frame_length * 120000 / 1_000.0))
    #hop_len_frames = int(round(frame_shift * 120000 / 1_000.0))
    #time_per_slize = ((256 - 1) * hop_len_frames + win_len_frames) / 120000
    #print(f"Time per slize of {256} is {time_per_slize} s of audio")


    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=128,
        low_freq=0.0,
        high_freq=0.0,
        dither=0.0,
        frame_shift=frame_shift,
        frame_length=frame_length,
    )

    fmin = fbank.min()
    fmax = fbank.max()
    spectrogram = (fbank - fmin) / (fmax - fmin + 1e-12)

    return spectrogram


class AudioIterableDataset(IterableDataset):
    def __init__(self, file_paths, transform=None, shuffle=False, classes=None, label_to_idx=None, multilabel=True, num_classes=0):
        super().__init__()
        self.file_paths = file_paths
        self.transform = transform
        self.shuffle = shuffle
        self.max_sample_rate= 120000

        self.classes = classes
        self.label_to_idx = label_to_idx
        self.multilabel = multilabel
        self.num_classes = num_classes

        if self.shuffle and isinstance(self.file_paths, pd.DataFrame):
            self.file_paths = self.file_paths.sample(frac=1, random_state=42).reset_index(drop=True)

    #Supports multilabeling through "specie1;specie2" lists or just single value
    def _encode_labels(self, raw):
        if self.multilabel:
            target = torch.zeros(len(self.classes), dtype=torch.float32)
            if isinstance(raw, str):
                labels = [s.strip() for s in re.split(r"[;,]", raw) if s.strip()]
            elif isinstance(raw, (list, tuple, set)):
                labels = list(raw)
            else:
                labels = [str(raw)]
            for lab in labels:
                idx = self.label_to_idx.get(lab)
                if idx is not None:
                    target[idx] = 1.0
            return target
        else:
            # one-hot via class index (use CrossEntropyLoss later)
            idx = self.label_to_idx[str(raw)]
            return torch.tensor(idx, dtype=torch.long)

    def compute_required_samples(self, desired_frames: int, sample_rate: int) -> int:
        hop = int(round(10 * sample_rate / 1000))  # 10 ms
        win = int(round(25 * sample_rate / 1000))  # 25 ms
        return (desired_frames - 1) * hop + win

    def _process_file(self, row):
        path = row['Path']
        label_begin_offset = row['Label Begin Offset']

        label = row['Species']
        label = self._encode_labels(label) #Multi label encoded

        #sample_rate = torchaudio.info(path).sample_rate #Super slow on slurm cluster, doubles i/o time
        sample_rate = row["Sample Rate"]

        if sample_rate > self.max_sample_rate:
            required_samples = self.compute_required_samples(1024, self.max_sample_rate)
            read_samples = int(math.ceil(required_samples * sample_rate / self.max_sample_rate))
            waveform, _ = torchaudio.load(
                path,
                frame_offset=int(sample_rate * label_begin_offset),
                num_frames=read_samples,
            )
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.max_sample_rate)
            sample_rate = self.max_sample_rate

        else:
            required_samples = self.compute_required_samples(1024, sample_rate)
            waveform, _ = torchaudio.load(
                path,
                frame_offset=int(sample_rate * label_begin_offset),
                num_frames=required_samples,
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        fbank = convert_numpy_to_fbank(waveform, sample_rate)
        fbank = fbank.transpose(0, 1)
        if fbank.size(1) < 1024:
            return
        if fbank.size(1) > 1024:
            fbank = fbank[:, :1024]

        if self.transform:
            fbank = self.transform(fbank)

        #Technically a waste that this is an audioIterable, legacy from pretraining. TODO change
        yield (fbank, label)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # num_workers=0
            file_iter = range(len(self.file_paths))
        else:  # Multi-worker
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.file_paths) / float(num_workers)))

            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_paths))
            file_iter = range(start, end)

        for idx in file_iter:
            row = self.file_paths.iloc[idx] if isinstance(self.file_paths, pd.DataFrame) else self.file_paths[idx]
            yield from self._process_file(row)


class DataloaderModule(LightningDataModule):
    def __init__(
            self,
            label_file_path: [str] = None,
            location_filter=None,
            species_filter=None,
            train_val_test_split: Tuple[float, float, float] = (1, 0, 0),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()
        self.label_file_path = label_file_path
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.image_transforms = transforms.Compose([
        ])


        self.df = pd.read_excel(label_file_path)

        # Filter for datasets to finetune on
        if location_filter is not None and location_filter != []:
            self.df = self.df.query("Location in @location_filter")
        if species_filter is not None and species_filter != []:
            self.df = self.df.query("Species in @species_filter")

        self.classes = sorted(self.df["Species"].dropna().unique().tolist())
        self.label_to_idx = {c: i for i, c in enumerate(self.classes)}
        print("Classes:", self.classes)
        self.num_classes = len(self.classes)

        # TODO check out the balancing of the datasets for validation

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None):
        """
        Split self.all_file_paths into train/val/test and create the
        corresponding IterableDatasets.
        """

        train_ratio, val_ratio, test_ratio = self.train_val_test_split  # (0.7,0.2,0.1)
        train_df = self.df.sample(frac=train_ratio, random_state=999)
        val_df = self.df.drop(train_df.index)

        # Now create one dataset per split
        self.data_train = AudioIterableDataset(
            file_paths=train_df,
            transform=self.image_transforms,
            shuffle=True,
            classes=self.classes,
            label_to_idx=self.label_to_idx,
            num_classes=self.num_classes,
        )
        self.data_val = AudioIterableDataset(
            file_paths=val_df,
            transform=self.image_transforms,
            shuffle=True,
            classes=self.classes,
            label_to_idx=self.label_to_idx,
            num_classes=self.num_classes,
        )
        # self.data_test = AudioIterableDataset(
        #    file_paths=test_paths,
        #    transform=self.image_transforms,
        #    shuffle=False
        # )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            prefetch_factor=1,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=3,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=0,
            persistent_workers=False,
        )


def show_np_spectrogram_file(waveform, file_name="", vmin=None, vmax=None):
    plt.figure(figsize=(10, 5))
    title = file_name
    # title = f"filename: {file_name}"# waveform min: {waveform.min()}, waveform max: {waveform.max()}"
    if vmax is not None:
        img = plt.imshow(
            waveform,
            cmap='turbo',
            interpolation='none',
            origin='lower',
            aspect='auto',
            vmax=vmax,
        )
    else:
        img = plt.imshow(
            waveform,
            cmap='turbo',
            interpolation='none',
            origin='lower',
            aspect='auto',
            # vmin=vmin, vmax=vmax
        )
    plt.ylim(0, waveform.shape[0])  # Frequency bins
    plt.xlim(0, waveform.shape[1])  # Time frames
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.colorbar(img, label="dB", pad=0.020)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{file_name}.png")
    #plt.show()


def test_dataloader(batch_size):
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
    #label_file = "C:\\Users\\ander\\Github\\Masters\\Pdata\\labels.xlsx"

    data_module = DataloaderModule(label_file_path=label_file, location_filter=location_filter, batch_size=batch_size, num_workers=8)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    for batch_idx, spectrograms in enumerate(train_loader):
        #print(f"Batch {batch_idx + 1}, Spectrograms Shape: {spectrograms.shape}")
        i = 0
        for spectrogram in spectrograms:
            assert spectrogram.shape == (128, 256), f"Got {spectrogram.shape}, expected (128, 256)"
            show_np_spectrogram_file(spectrogram, f"num: {i}")
            i+=1
        break


def extract_segment(spectrogram, sample_rate, hop_length, start_time, end_time):
    start_sample = int(sample_rate * start_time)
    end_sample = int(sample_rate * end_time)

    start_frame = start_sample // hop_length
    end_frame = end_sample // hop_length

    return spectrogram[:, start_frame:end_frame]


def visualize_waveform(file_paths, fbank_processor, postfix_name):
    for file_path in file_paths:
        waveform, sr = torchaudio.load(file_path)
        fbank, sr = fbank_processor(waveform=waveform, sample_rate=sr)  # shape: [n_mels, time]
        print(fbank.shape)
        fbank = extract_segment(fbank, sr, fbank_processor.hop_len, 7, 12)
        print(fbank.shape)
        show_np_spectrogram_file(fbank, file_name=f"{file_path} - {postfix_name}.png")

def main():
    os.environ[
        "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # No idea what causes the error when this is removed, since all its common causes is not present, only runs when testing
    # visuaziation_settings_test()
    test_dataloader(batch_size=16)


if __name__ == "__main__":
    main()
