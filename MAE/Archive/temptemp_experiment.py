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

#AudioMAE baseline
def convert_numpy_to_fbank(waveform, sample_rate):
    """
    Converts a numpy array (audio segment) to a mel spectrogram using torchaudio's fbank function.
    """
    max_sample_rate = 120000
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

# pip install nnAudio
import math
import torch
import torchaudio
from nnAudio.features.cqt import CQT2010v2

class CQTProcessor:
    """
    Constant-Q spectrogram using nnAudio's CQT1992v2 (log-frequency bins).
    Keeps control over hop/window (temporal overlap) and exact #bins.
    """
    def __init__(
        self,
        win_len_ms: float = 25.0,     # only used if you still want to define hop from ms
        hop_len_ms: float = 10.0,
        max_sample_rate: int = 120_000,
        n_bins: int = 128,            # total CQT bins you want
        bins_per_octave: int = 12,    # resolution per octave
        f_min: float = 30,          # C1 ~ 32.7 Hz, pick what suits your data
        power_to_db_top: float = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.win_len_ms = win_len_ms
        self.hop_len_ms = hop_len_ms
        self.max_sample_rate = max_sample_rate
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.f_min = f_min
        self.power_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=power_to_db_top
        ).to(device)
        self.device = device
        self.cqt = None  # will be built lazily once we know sr/hop_len

    @staticmethod
    def _ms_to_samples(ms: float, sr: int) -> int:
        return int(round(ms * sr / 1_000.0))

    def _build_cqt(self, sr: int):
        hop_len = self._ms_to_samples(self.hop_len_ms, sr)
        # nnAudio's CQT uses hop_length directly; "win_len" is internally determined per bin.
        self.cqt = CQT2010v2(
            sr=sr,
            fmin=self.f_min,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            hop_length=hop_len,
            window='hann',
            # center=False more closely matches your STFT code; nnAudio defaults center=True
            # but CQT1992v2 forces center=True internally; it's usually fine.
            # pad_mode='reflect',  # optional; default is "reflect"
            filter_scale=0.5,
            norm=True,
            verbose=False,
            output_format='Magnitude',  # "Magnitude" or "Complex"
        ).to(self.device)
        return hop_len

    def __call__(self, waveform: torch.Tensor, sample_rate: int):
        waveform = waveform.to(self.device)
        waveform = waveform - waveform.mean()

        if sample_rate > self.max_sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.max_sample_rate
            ).to(self.device)(waveform)
            sample_rate = self.max_sample_rate

        if self.cqt is None:
            hop_len = self._build_cqt(sample_rate)
        else:
            hop_len = self.cqt.hop_length

        # nnAudio expects (batch, time). If mono (1, T), squeeze first dim.
        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)
        # Now shape is (T,) or (channels, T). nnAudio handles (batch, time).
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # (batch, n_bins, time)
        cqt_mag = self.cqt(waveform)

        # Convert magnitude -> power (square) then to dB
        spec_power = cqt_mag ** 2
        spec_db = self.power_to_db(spec_power)

        # Normalize 0-1
        spec_db = (spec_db - spec_db.amin(dim=[1,2], keepdim=True)) / (
            spec_db.amax(dim=[1,2], keepdim=True) - spec_db.amin(dim=[1,2], keepdim=True) + 1e-10
        )

        # Return first (mono) channel if you had 1 input
        spec_db = spec_db.squeeze(0)  # (n_bins, time)

        win_len = None  # CQT doesn't have a single fixed win_len
        return spec_db, sample_rate, win_len, hop_len



def prosess_file_modern(path, frame_offset=0, num_frames=-1):
    n_freq_bins = 128
    #n_fft = 4096
    #win_len = 3000#4096
    #hop_len = 1200#win_len // 4  # 50% overlap
    win_len_ms = 25#15#34.13
    hop_len_ms = 10#6#17.07


    frames_per_segment = 512

    fbank_processor = CQTProcessor()

    waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    fbank, sr, win_len, hop_len = fbank_processor(waveform=waveform, sample_rate=sr)

    #time_per_slize = ((frames_per_segment - 1) * hop_len + win_len) / sr
    #print(f"Time per slize of {frames_per_segment} is {time_per_slize} s of audio")

    pretrain_dataloader.show_np_spectrogram_file(fbank.cpu().T, file_name=f"Modern Linear {path}", vmax=1)

    slices = pretrain_dataloader.slice_spectrogram(fbank, sr, 16, 0.2, hop_len, frames_per_segment=frames_per_segment)

    i = 0
    for slice in slices:
        pretrain_dataloader.show_np_spectrogram_file(slice.cpu(),
                                                     file_name=f"Modern {path}",vmax=1)  # =path.split("\\")[-1],vmax=1, title="Modern")
        if i > 10:
            break
        i +=1


def prosess_file_like_audiomae(path, frame_offset=0, num_frames=-1):
    waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    fbank = convert_numpy_to_fbank(waveform, sr)
    fbank = fbank.transpose(0, 1)
    slices = fbank.unfold(dimension=1, size=512, step=512)
    slices = torch.permute(slices, (1, 2, 0))
    pretrain_dataloader.show_np_spectrogram_file(fbank.T,
                                                 file_name=f"AudioMAE {path}")

    win_len = int(round(25 * 120000 / 1_000.0))
    hop_len = int(round(10 * 120000 / 1_000.0))
    time_per_slize = ((512 - 1) * hop_len + win_len) / sr
    print(f"Time per slize of {512} is {time_per_slize} s of audio")

    i = 0
    for slice in slices:
        pretrain_dataloader.show_np_spectrogram_file(slice,
                                                     file_name="AudioMAE")  # file_name=path.split("\\")[-1], title="AudioMAE")
        if i > 10:
            break
        i +=1

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
        #"C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\CMHS_2022_05_09_12_22_37.wav", # Sperm whale 10kHz
        "C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\2019_07_04_10_53_34.wav",  # Dolphin 69kHz
    ]
    for path in paths:
        #prosess_file_like_audiomae(path)
        prosess_file_modern(path)


def main():
    os.environ[
        "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # No idea what causes the error when this is removed, since all its common causes is not present, only runs when testing
    show_high_freq_files()
    # show_low_freq_labeled()


if __name__ == "__main__":
    main()
