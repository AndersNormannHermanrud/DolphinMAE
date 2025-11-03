import math
import pathlib
import random
import sys
import warnings
from argparse import ArgumentParser
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
# import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, IterableDataset
# from util import show_np_spectrogram_file
# from torchvision import transforms
# from pytorch_lightning import LightningDataModule
import os
import pretrain_dataloader


def convert_numpy_to_fbank(waveform, sample_rate):
    """
    Converts a numpy array (audio segment) to a mel spectrogram using torchaudio's fbank function.
    """
    max_sample_rate = 140000
    if sample_rate > max_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=max_sample_rate)
        sample_rate = max_sample_rate
        waveform = resampler(waveform)

    waveform = waveform - waveform.mean()
    frame_length = 25
    frame_shift = 10
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
    return fbank


class FbankProcessor:
    def __init__(self, sample_rate=None, n_freq_bins=128, n_fft=1024, win_len=25, hop_len=10):
        """
        A reusable transform that replaces `torchaudio.compliance.kaldi.fbank`.
        """
        self.n_freq_bins = n_freq_bins
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.max_sample_rate = 140000  # TODO see towards sample rate range
        # TODO Ask biologist about this
        # For each species, is the amount of calls average around the most labeled dataset (Blivenichia) but just not labeled.

        self.log_mel_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=100)

        warnings.warn(f"UserWarning: downsamling all samples above {self.max_sample_rate} down to this sr")

    def transform(self, waveform, sample_rate):
        """
        This function is wildly inefficient as it creates a new audio transformer object each time it converts, but this is a workaround to the files having different sampling rate
        TODO create one instance for each sr, or better practice resample all files to same sr
        """
        audio_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=self.n_freq_bins,
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop_len,
            window_fn=torch.hann_window,
            center=False,
            power=2.0,
            norm="slaney",
            f_min=20.0,
            mel_scale="slaney"
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


def prosess_file_modern(path, frame_offset=0, num_frames=-1):
    n_freq_bins = 128
    n_fft = 4096
    win_len = n_fft  # 4096
    hop_len = win_len // 2  # 50% overlap

    frames_per_segment = 512

    fbank_processor = FbankProcessor(sample_rate=None, n_freq_bins=n_freq_bins, n_fft=n_fft,
                                     win_len=win_len,
                                     hop_len=hop_len)

    waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    fbank, sr = fbank_processor(waveform=waveform, sample_rate=sr)

    time_per_slize = ((frames_per_segment - 1) * hop_len + win_len) / sr
    print(f"Time per slize of {frames_per_segment} is {time_per_slize} s of audio")

    pretrain_dataloader.show_np_spectrogram_file(fbank.T, file_name=f"Modern LogMel {path}")

    #slices = pretrain_dataloader.slice_spectrogram(fbank, sr, 16, 0.2, hop_len, frames_per_segment=frames_per_segment)
    #for slice in slices:
    #    pretrain_dataloader.show_np_spectrogram_file(slice,
    #                                                 file_name="Modern")  # =path.split("\\")[-1],vmax=1, title="Modern")


def prosess_file_like_audiomae(path, frame_offset=0, num_frames=-1):
    waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    fbank = convert_numpy_to_fbank(waveform, sr)
    fbank = fbank.transpose(0, 1)
    slices = fbank.unfold(dimension=1, size=1024, step=824)
    slices = torch.permute(slices, (1, 2, 0))
    pretrain_dataloader.show_np_spectrogram_file(fbank.T,
                                                 file_name=f"AudioMAE {path}")
    #for slice in slices:
    #    pretrain_dataloader.show_np_spectrogram_file(slice,
    #                                                 file_name="AudioMAE")  # file_name=path.split("\\")[-1], title="AudioMAE")


def show_low_freq_labeled():
    path = "C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\170927_070000_AU_AT04.wav"
    metadata = torchaudio.info(path)
    start_time_sec = 180
    end_time_sec = 250

    frame_offset = int(start_time_sec * metadata.sample_rate)
    num_frames = int((end_time_sec - start_time_sec) * metadata.sample_rate)

    # prosess_file_like_audiomae(path, frame_offset=frame_offset, num_frames=num_frames)
    prosess_file_modern(path, frame_offset=frame_offset, num_frames=num_frames)


def show_high_freq_files():
    paths = [  # "C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\CMHS_2020_08_19_11_34_01.wav",
        # "C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\CMHS_2022_05_09_12_17_31.wav",
        "C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\CMHS_2022_05_09_12_22_37.wav", # Sperm whale 10kHz
        "C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\2019_07_04_10_53_34.wav",  # Dolphin 69kHz
    ]
    for path in paths:
        prosess_file_like_audiomae(path)
        prosess_file_modern(path)


def main():
    os.environ[
        "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # No idea what causes the error when this is removed, since all its common causes is not present, only runs when testing
    show_high_freq_files()
    # show_low_freq_labeled()


if __name__ == "__main__":
    main()
