import math
import pathlib
import random
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, IterableDataset
# from util import show_np_spectrogram_file
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import os


class FbankProcessor:
    def __init__(self, sample_rate=None, n_freq_bins=128, n_fft=1024, win_len_ms=25, hop_len_ms=10):
        """
        A reusable transform that replaces `torchaudio.compliance.kaldi.fbank`.
        """
        self.n_freq_bins = n_freq_bins
        self.n_fft = n_fft
        self.win_len_ms = win_len_ms
        self.hop_len_ms = hop_len_ms
        self.max_sample_rate = 140000  # TODO see towards sample rate range
        # TODO Ask biologist about this
        # For each species, is the amount of calls average around the most labeled dataset (Blivenichia) but just not labeled.

        self.log_mel_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=100)

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
            n_mels=self.n_freq_bins,
            n_fft=self.n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            center=False,
            power=2.0,
            norm="slaney",
            f_min=4.0,
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

        # Normalize fbank
        spectrogram = spectrogram.numpy()
        spectrogram = torch.from_numpy(spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

        return spectrogram, sample_rate


def slice_spectrogram(
        fbank: torch.Tensor,
        sample_rate,
        target_length_seconds,
        overlap_percentage=0.0,
        hop_length=10.0,
        frames_per_segment=1024,
):
    # fbank = fbank.transpose(0, 1)  # shape [T, n_mels]
    if fbank.shape[1] < 1000:
        return []

    #frames_per_segment = 1024  # int((target_length_seconds * sample_rate) / hop_length)
    if frames_per_segment <= 0:
        raise ValueError("target_length_seconds too small or frame_shift_ms too large.")

    step_size = int(frames_per_segment * (1 - overlap_percentage))
    step_size = max(step_size, 1)

    slices = fbank.unfold(dimension=1, size=frames_per_segment, step=step_size)
    slices = torch.permute(slices, (1, 2, 0))

    return slices


class AudioIterableDataset(IterableDataset):
    def __init__(self, file_paths, transform=None, shuffle=False):
        super().__init__()
        self.file_paths = file_paths
        self.transform = transform
        self.shuffle = shuffle

        n_freq_bins = 128
        n_fft = 4096
        win_len_ms = 4096
        self.hop_len_ms = 4096 // 2
        self.fbank_processor = FbankProcessor(sample_rate=None, n_freq_bins=n_freq_bins, n_fft=n_fft,
                                              win_len_ms=win_len_ms,
                                              hop_len_ms=self.hop_len_ms)

        if self.shuffle:
            random.shuffle(self.file_paths)

    def _process_file(self, path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        fbank, sr = self.fbank_processor(waveform=waveform, sample_rate=sr)  # shape: [n_mels, time]
        slices = slice_spectrogram(fbank, sr, 16, 0.2, self.hop_len_ms)
        if len(slices) == 0:
            print(
                f"Warning, file {path} is too short, consider removing it from the dataset. If its many files, consider adjusting spectrogram settings")
            return

        if self.transform:
            slices = [self.transform(s) for s in slices]

        slices = [s.unsqueeze(0) for s in slices]

        # Debug. and all 0 spectrogram could theoretically return inf or -inf values
        # try:
        #    slices_tensor = torch.tensor(slices, dtype=torch.float32)
        #    if not torch.isfinite(slices_tensor).all():
        #        print(f"[Warning] Non-finite values in slices from file: {path}")
        # except Exception as e:
        #    print(f"[Error] Could not check slices for file {path}: {e}")

        for spec_slice in slices:
            yield spec_slice

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
            path = self.file_paths[idx]
            yield from self._process_file(path)


class DataloaderModule(LightningDataModule):
    def __init__(
            self,
            data_dirs: [str] = None,
            train_val_test_split: Tuple[float, float, float] = (1, 0, 0),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.image_transforms = transforms.Compose([
        ])

        # Note, this works as long as all paths can be kept in memory, for a larger (millions) dataset, switch to iterating over a path file
        self.all_file_paths = []
        for folder in self.data_dirs:
            i = 0
            for f in os.listdir(folder):
                if f.endswith(".wav"):
                    self.all_file_paths.append(os.path.join(folder, f))
                    i += 1
            print(f"{i} file path loaded from {folder}")
        # self.all_file_paths = np.array(self.all_file_paths)
        print(f"Found {len(self.all_file_paths)} total .wav files.")
        # Note, I am shuffling here manually instead of the dataloader
        random.shuffle(self.all_file_paths)

        # TODO check out the balancing of the datasets for validation

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None):
        """
        Split self.all_file_paths into train/val/test and create the
        corresponding IterableDatasets.
        """

        total_len = len(self.all_file_paths)
        train_ratio, val_ratio, test_ratio = self.train_val_test_split
        train_size = int(train_ratio * total_len)
        val_size = int(val_ratio * total_len)
        # test_size = total_len - train_size - val_size

        train_paths = self.all_file_paths[:train_size]
        val_paths = self.all_file_paths[train_size: train_size + val_size]
        test_paths = self.all_file_paths[train_size + val_size:]

        # Now create one dataset per split
        self.data_train = AudioIterableDataset(
            file_paths=train_paths,
            transform=self.image_transforms,
            shuffle=False
        )
        self.data_val = AudioIterableDataset(
            file_paths=val_paths,
            transform=self.image_transforms,
            shuffle=False
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
            prefetch_factor=3,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
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
    plt.figure(figsize=(25, 5))
    title = file_name
    # title = f"filename: {file_name}"# waveform min: {waveform.min()}, waveform max: {waveform.max()}"
    if vmax is not None:
        img = plt.imshow(
            waveform.T,
            cmap='turbo',
            interpolation='none',
            origin='lower',
            aspect='auto',
            vmax=vmax,
        )
    else:
        img = plt.imshow(
            waveform.T,
            cmap='turbo',
            interpolation='none',
            origin='lower',
            aspect='auto',
            # vmin=vmin, vmax=vmax
        )
    plt.ylim(0, waveform.T.shape[0])  # Frequency bins
    plt.xlim(0, waveform.T.shape[1])  # Time frames
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.colorbar(img, label="dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def test_dataloader(batch_size):
    data_dirs = [
        "/cluster/projects/uasc/Datasets/Data univ Bari/dataset",
        "/cluster/projects/uasc/Datasets/Dubrovnik/dataset",
        "/cluster/projects/uasc/Datasets/Fram Strait 2008-09/dataset",
        "/cluster/projects/uasc/Datasets/Blitvenica/dataset",
        "/cluster/projects/uasc/Datasets/Fram Strait 2017-18/dataset",
        "/cluster/projects/uasc/Datasets/Atwain 2017-18/dataset",
        "/cluster/projects/uasc/Datasets/Losinj/dataset",
    ]
    data_dirs = ["C:\\Users\\ander\\OneDrive\\Masters\\Pdata"]
    data_module = DataloaderModule(data_dirs=data_dirs, batch_size=batch_size, num_workers=1)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    for batch_idx, spectrograms in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}, Spectrograms Shape: {spectrograms.shape}")
        # assert spectrograms.shape != (128, 937), f"This sample is not the right size"
        for spectrogram in spectrograms:
            show_np_spectrogram_file(spectrogram, file_name="spectrogram.png")


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


def visuaziation_settings_test():
    data_dirs = ["C:\\Users\\ander\\Github\\Masters\\Pdata\\CMHS_2022_05_09_12_17_31.wav"]
    n_mels_a = [128]
    n_fft_a = [4096]  # 16384  # 1024
    win_len_ms_a = [4096]
    hop_len_ms_a = [4096 // 4]
    for n_mels, n_fft, win_len_ms, hop_len_ms in zip(n_mels_a, n_fft_a, win_len_ms_a, hop_len_ms_a):
        # all_file_paths = [f"n_mels {n_mels}, n_fft {n_fft}, win_len {win_len}, hop_len {hop_len}" for f in all_file_paths]
        fbank_processor = FbankProcessor(sample_rate=None, n_mels=n_mels, n_fft=n_fft, win_len_ms=win_len_ms,
                                         hop_len_ms=hop_len_ms)
        visualize_waveform(data_dirs, fbank_processor,
                           f"n_mels {n_mels}, n_fft {n_fft}, win_len_ms {win_len_ms}, hop_len_ms {hop_len_ms}")


def main():
    os.environ[
        "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # No idea what causes the error when this is removed, since all its common causes is not present, only runs when testing
    # visuaziation_settings_test()
    test_dataloader(batch_size=1)


if __name__ == "__main__":
    main()
