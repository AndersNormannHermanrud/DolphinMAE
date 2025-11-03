import math
import pathlib
import random
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from matplotlib import pyplot as plt
from torch.optim.optimizer import required
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import os


class AudioIterableDataset(IterableDataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform

        n_mels = 128
        n_fft = 4096  # 16384  # 1024
        self.win_len = 4096
        self.hop_len = 4096 // 2
        self.max_sample_rate = 140000
        self.fbank_processor = FbankProcessor(sample_rate=None, n_mels=n_mels, n_fft=n_fft, win_len_ms=self.win_len,
                                              hop_len_ms=self.hop_len, max_sample_rate=self.max_sample_rate)

    def compute_required_samples(self, desired_frames: int) -> int:
        num_samples = (desired_frames - 1) * self.hop_len + self.win_len
        print(f"returning {num_samples} samples")
        return num_samples

    def _process_file(self, row):
        path = row['Path']
        #location = row['Location']
        #species = row['Species']
        #low_freq = row['Low Freq']
        #high_freq = row['High Freq']
        label_begin_offset = row['Label Begin Offset']
        #label_end_offset = row['Label End Offset']
        #label_duration = row['Label Duration']
        #recording_begin_time = row['Recording Begin Time']
        #recording_end_time = row['Recording End Time']
        #recording_duration = row['Recording Duration']
        #sample_rate = row['Sample Rate']

        # TODO Load correctly
        #waveform, _ = torchaudio.load(
        #    path,
        #    frame_offset=int(sample_rate * label_begin_offset),
        #    num_frames=int(sample_rate * 10)
        #)
        sample_rate = torchaudio.info(path).sample_rate
        required_samples = self.compute_required_samples(1024)
        if sample_rate > self.max_sample_rate:
            waveform, _ = torchaudio.load(
                path,
                frame_offset=int(sample_rate * label_begin_offset),
            )
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.max_sample_rate)
            waveform = waveform[:required_samples]
        else:
            waveform, _ = torchaudio.load(
                path,
                frame_offset=int(sample_rate * label_begin_offset),
                num_frames=required_samples,
            )

        if waveform.shape[1] < required_samples:
            print(f"[WARN] Not enough samples. Got {waveform.shape[1]}, expected {required_samples}.")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        fbank, sr = self.fbank_processor(waveform=waveform, sample_rate=sample_rate)
        # TODO redo the slizing, logic to slize if it is large enough?
        # slices = slice_spectrogram(fbank, sr, 16, 0.2, self.hop_len)

        if self.transform:
            fbank = self.transform(fbank)

        yield fbank

    def __iter__(self):
        """
        Each worker will run this to produce items. We partition self.file_paths
        among workers to avoid duplicating work.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # num_workers=0
            df_iterate = self.df
        else:  # Multi-worker
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.df.index) / float(num_workers)))

            start = worker_id * per_worker
            end = min(start + per_worker, len(self.df.index))
            df_iterate  = self.df.iloc[start:end]

        for idx, row in df_iterate.iterrows():
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

        # Filter for datasets to finetune on ["Data univ Bari","Losinj"]
        if location_filter is not None and location_filter != []:
            self.df = self.df.query("Location in @locations_to_train_on")
        if species_filter is not None and species_filter != []:
            self.df = self.df.query("Location in @species_filter")

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
        # test_df = val_df.sample(frac=test_ratio, random_state=999)

        self.data_train = AudioIterableDataset(
            df=train_df,
            transform=self.image_transforms,
        )
        self.data_val = AudioIterableDataset(
            df=val_df,
            transform=self.image_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2
        )


class FbankProcessor:
    def __init__(self, sample_rate=None, n_mels=128, n_fft=1024, win_len_ms=25, hop_len_ms=10, max_sample_rate=100000):
        """
        A reusable transform that replaces `torchaudio.compliance.kaldi.fbank`.
        """
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_len_ms = win_len_ms
        self.hop_len_ms = hop_len_ms
        self.max_sample_rate = max_sample_rate  # TODO see towards sample rate range
        # TODO Ask biologist about this
        # For each species, is the amount of calls average around the most labeled dataset (Blivenichia) but just not labeled.

        self.log_mel_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

        print(f"Warning, downsamling all samples above {self.max_sample_rate} down to this sr")

    def transform(self, waveform, sample_rate):
        """
        This function is wildly inefficient as it creates a new audio transformer object each time it converts, but this is a workaround to the files having different sampling rate
        TODO create one instance for each sr, or better practice resample all files to same sr
        """
        # Conversion from ms to samples
        win_length = self.win_len_ms  # min(self.n_fft, int(sample_rate * self.win_len / 1000.0))
        hop_length = self.hop_len_ms  # int(sample_rate * self.hop_len / 1000.0)
        audio_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            center=False,
            power=2.0,
            norm="slaney",
            # mel_scale="slaney"
        )
        waveform = audio_transform(waveform)
        return waveform

    def __call__(self, waveform: torch.Tensor, sample_rate: int):
        waveform -= waveform.mean()

        if sample_rate > self.max_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.max_sample_rate)
            sample_rate = self.max_sample_rate
            waveform = resampler(waveform)

        spectrogram = self.transform(waveform, sample_rate=sample_rate)
        spectrogram = self.log_mel_transform(spectrogram)
        spectrogram = spectrogram.squeeze(0)
        return spectrogram, sample_rate


def slice_spectrogram(
        fbank: torch.Tensor,
        sample_rate: int,
        target_length_seconds: float,
        overlap_percentage: float = 0.0,
        hop_length: float = 10.0
):
    # fbank = fbank.transpose(0, 1)  # shape [T, n_mels]
    if fbank.shape[1] < 1000:
        return []

    frames_per_segment = 1024  # int((target_length_seconds * sample_rate) / hop_length)
    if frames_per_segment <= 0:
        raise ValueError("target_length_seconds too small or frame_shift_ms too large.")

    step_size = int(frames_per_segment * (1 - overlap_percentage))
    step_size = max(step_size, 1)

    slices = fbank.unfold(dimension=1, size=frames_per_segment, step=step_size)
    slices = torch.permute(slices, (1, 2, 0))

    return slices


def show_np_spectrogram_file(waveform, file_name=""):
    plt.figure(figsize=(25, 5))
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
    plt.title(file_name)
    plt.show()


def test_dataloader(batch_size):
    data_dirs = "/cluster/projects/uasc/Datasets/labels.xlsx"
    #data_dirs = "C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\labels.xlsx"
    data_module = DataloaderModule(label_file_path=data_dirs, batch_size=batch_size, num_workers=16)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    unique_shapes = []

    #print(data_module.df.info())
    #print(data_module.df.head())

    for batch_idx, spectrograms in enumerate(train_loader):
        if batch_idx > 100:
            break
        print(f"Batch {batch_idx + 1}, Spectrograms Shape: {spectrograms.shape}")
        for spectrogram in spectrograms:
            # show_np_spectrogram_file(spectrogram, file_name="spectrogram.png")
            if spectrogram.shape not in unique_shapes:
                unique_shapes.append(spectrogram.shape)

    highest_x_touple = max(unique_shapes, key=lambda i: i[1])
    lowest_x_touple = min(unique_shapes, key=lambda i: i[1])
    average_x = sum(shape[1] for shape in unique_shapes) / len(unique_shapes)
    unique_shapes = np.array(unique_shapes)
    percentile_30 = np.percentile(unique_shapes[:, 1], 30)
    percentile_70 = np.percentile(unique_shapes[:, 1], 70)

    print(f"highest x touple {highest_x_touple}")
    print(f"lowest x touple {lowest_x_touple}")
    print(f"average x {average_x}")
    print(f"percentile_30 {percentile_30}")
    print(f"percentile_70 {percentile_70}")

    plt.scatter(x=unique_shapes[:, 1], y=unique_shapes[:, 0], color='green', label='Shape')
    plt.title("Sizes")
    plt.legend()
    # plt.show()
    plt.savefig("Sizes.png", format='png')


    print("All samples loaded and is the right shape")

def main():
    os.environ[
        "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # No idea what causes the error when this is removed, since all its common causes is not present, only runs when testing
    # visuaziation_settings_test()
    test_dataloader(batch_size=1)


if __name__ == "__main__":
    main()
