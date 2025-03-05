import collections
import itertools
import random
import sys
import time
import traceback
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler, Sampler
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchaudio.transforms import Spectrogram
from PIL import Image
import os
import librosa


class Dataloader(Dataset):
    def __init__(self, data_dirs: [str], transform=None):
        super().__init__()
        self.data_dirs = data_dirs
        self.file_paths = []
        self.transform = transform
        self.queue = collections.deque()

        for folder in self.data_dirs:
            i = 0
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    self.file_paths.append(os.path.join(folder, file))
                    i += 1
            print(f"{i} file path loaded from {folder}")
        print(f"{len(self.file_paths)} file path loaded in total from {len(self.data_dirs)} folders")

    def get_from_queue(self):
        """Returns an item from the queue without calling __getitem__"""
        if len(self.queue) > 0:
            return self.queue.popleft()
        return None  # Queue is empty

    def get_queue_length(self):
        return len(self.queue)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return None

    def load(self, idx):
        path = self.file_paths[idx]
        print(f"Loading {path}")
        audio_samples, sr = librosa.load(path, sr=None)
        spectrogram = convert_numpy_to_fbank(audio_samples, sr)
        spectrogram = slice_spectrogram(spectrogram, 10, 0.5, 10)
        spectrogram = [s.clone().detach().permute(1, 0) for s in spectrogram]
        sys.stdout.flush()

        if self.transform:
            spectrogram = [self.transform(s) for s in spectrogram]

        # self.queue.extend(spectrogram)
        return spectrogram

"""
This class is used when we make subsets of our dataset. this is the class loaded into the sampler
"""
class SubsetDatasetWithQueue(Dataset):
    def __init__(self, dataset, indices):
        """Wraps a subset while preserving queue operations."""
        self.dataset = dataset
        self.indices = indices
        self.queue = collections.deque()
        #for i in range(100):
        #    self.queue.append(i)

    def __len__(self):
        return len(self.indices)

    def get_from_queue(self, num):
        """Returns an item from the queue without calling __getitem__"""
        #print(num)
        #if len(self.queue) < num:
        #    raise IndexError(f"Not enough items in queue {self.get_from_queue()} < {num}")

        #samples = []
        # for i in range(num):
        samples = self.queue.popleft()
        print(f"Queue Length: {self.get_queue_length()}")
        return samples
        # return None  # Queue is empty

    def get_queue_length(self):
        return len(self.queue)

    def __getitem__(self, idx):
        return self.get_from_queue(idx)

    def load(self, idx):
        """Uses dataset's logic while respecting subset indices."""
        idx = self.indices[idx]  # Convert subset index to actual dataset index
        spectrogram = self.dataset.load(idx)  # ✅ Calls parent dataset's `__getitem__`
        self.queue.extend(spectrogram)
        return spectrogram


class QueueAwareSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))  # Store dataset indices

        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        """Sampler waits for queue to be full before providing new indices."""

        while self.indices or self.dataset.get_queue_length() >= self.batch_size:
            if self.dataset.get_queue_length() >= self.batch_size:
                yield list(range(self.batch_size))  # ✅ Ensures PyTorch gets valid indices

            else:
                file_idx = self.indices.pop(0)
                _ = self.dataset.load(file_idx)


class BufferedBatchCollator:
    def __init__(self, batch_size, dataset):
        self.batch_size = batch_size
        self.dataset = dataset  # Pass dataset to the collator

    def __call__(self, batch):
        return None

    def load(self, batch):
        """Collects samples from dataset queue until batch size is reached."""
        spectrograms = []
        while len(spectrograms) < self.batch_size:
            sample = self.dataset.get_from_queue()
            if sample is not None:
                spectrograms.append(sample)
        return torch.stack(spectrograms)



"""
class BufferedBatchCollator:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.buffer = collections.deque()

    def __call__(self, batch):
        spectrograms = []

        while len(spectrograms) < self.batch_size and self.buffer:
            spectrograms.append(self.buffer.popleft())

        for spectrogram_segments in batch:
            for s in spectrogram_segments:
                if len(spectrograms) < self.batch_size:
                    spectrograms.append(s)
                else:
                    self.buffer.append(s)

        if len(spectrograms) > self.batch_size:
            # Put the extras back into the queue
            while len(spectrograms) > self.batch_size:
                self.buffer.appendleft(spectrograms.pop())

        return torch.stack(spectrograms)
"""


class DataloaderModule(LightningDataModule):
    def __init__(
            self,
            data_dirs: [str] = None,
            train_val_test_split: Tuple[float, float, float] = (1, 0.0, 0.0),
            batch_size: int = 32,
            num_workers: int = 1,
            pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dirs = data_dirs
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Define image transformations
        self.image_transforms = transforms.Compose([
            # transforms.ToTensor(), # Do not use
            # transforms.Resize((128, 128)),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        # self.collate_fn = BufferedBatchCollator(batch_size=self.batch_size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        full_dataset = Dataloader(
            data_dirs=self.data_dirs,
            transform=self.image_transforms,
        )

        train_size = int(self.train_val_test_split[0] * len(full_dataset))
        val_size = int(self.train_val_test_split[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        #train_set, val_set, test_set = torch.utils.data.random_split(
        #    full_dataset, [train_size, val_size, test_size]
        #)

        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            range(len(full_dataset)), [train_size, val_size, test_size]
        )

        # ✅ Wrap subsets in `SubsetDatasetWithQueue`
        self.data_train = SubsetDatasetWithQueue(full_dataset, train_indices)
        self.data_val = SubsetDatasetWithQueue(full_dataset, val_indices)
        self.data_test = SubsetDatasetWithQueue(full_dataset, test_indices)

        # self.data_train = train_set
        # self.data_val = val_set
        # self.data_test = test_set

    def train_dataloader(self):
        sampler = QueueAwareSampler(self.data_train, batch_size=self.batch_size, shuffle=True)
        #collate_fn = BufferedBatchCollator(self.batch_size, self.data_train)

        return DataLoader(
            self.data_train,
            # batch_size=self.batch_size,
            batch_sampler=sampler,
            num_workers=1,  # self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            # shuffle=True,
            # collate_fn=collate_fn,
        )

    def val_dataloader(self):
        sampler = QueueAwareSampler(self.data_val, batch_size=self.batch_size, shuffle=True)
        collate_fn = BufferedBatchCollator(self.batch_size, self.data_train)

        return DataLoader(
            self.data_train,
            # batch_size=self.batch_size,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            # shuffle=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        sampler = QueueAwareSampler(self.data_test, batch_size=self.batch_size, shuffle=True)
        collate_fn = BufferedBatchCollator(self.batch_size, self.data_train)

        return DataLoader(
            self.data_train,
            # batch_size=self.batch_size,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
            # shuffle=True,
            collate_fn=collate_fn,
        )


def convert_numpy_to_fbank(numpy_array, sample_rate):
    """
    Converts a numpy array (audio segment) to a mel spectrogram using torchaudio's fbank function.
    """
    n_mels = 128
    n_fft = 1024
    hop_length = None

    waveform = torch.from_numpy(numpy_array.copy()).float()  # Weird error when I do not create a copy of the tensor
    waveform = waveform - waveform.mean()
    waveform = waveform.unsqueeze(0)
    frame_length = 25
    frame_shift = 10
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=n_mels,
        low_freq=0.0,
        high_freq=0.0,
        dither=0.0,
        frame_shift=frame_shift,
        frame_length=frame_length,
    )
    return fbank


def slice_spectrogram(
        fbank,
        target_length_seconds,
        overlap_percentage=0.0,
        frame_shift_ms=10.0
):
    """
    Splits a full FBank spectrogram into segments corresponding to
    'target_length_seconds' of audio, with 'overlap_percentage'.

    Notes:
      - torchaudio.compliance.kaldi.fbank uses a frame_shift of 'frame_shift_ms'.
        e.g., 10 ms -> 100 frames per second.

      - We'll discard any leftover frames at the end if they
        don't fit a full segment.

    :param fbank: Full spectrogram, shape [num_frames, n_mels]
    :param target_length_seconds: How many *seconds* each slice should represent
    :param overlap_percentage: e.g. 0.0 = no overlap, 0.5 = 50% overlap
    :param frame_shift_ms: The frame shift used in the FBank (default 10 ms)
    """

    # Calculate how many FBank frames correspond to target_length_seconds.
    frames_per_second = 1000.0 / frame_shift_ms
    frames_per_segment = int(target_length_seconds * frames_per_second)
    if frames_per_segment <= 0:
        raise ValueError("target_length_seconds too small or frame_shift_ms too large.")

    # Overlap logic
    step_size = int(frames_per_segment * (1 - overlap_percentage))
    step_size = max(step_size, 1)

    start_frame = 0
    segments = []
    while (start_frame + frames_per_segment) <= fbank.shape[
        0]:  # NB dropping last segment of a file if shorter that desired length
        end_frame = start_frame + frames_per_segment
        segment = fbank[start_frame:end_frame, :]  # shape: [frames_per_segment, n_mels]

        # file_path = "/cluster/projects/uasc/Datasets/Data univ Bari/dataset/" + file_path.split("\\")[-1].replace("CMHS_", "").replace("TARANTO_","") # For testing on my own computer, artificially adjusting path
        # file_path = file_path.replace("Data univ Bari", "Data univ Bari copy")  # There is a naming error in the file compared to the original files

        segments.append(segment)
        start_frame += step_size
    return segments


def test_dataloader(batch_size: int = 16):
    # Create FewShotDataset instance
    data_dirs = [
        "/cluster/projects/uasc/Datasets/Data univ Bari/dataset",
        "/cluster/projects/uasc/Datasets/Dubrovnik/dataset",
        "/cluster/projects/uasc/Datasets/Fram Strait 2008-09/dataset",
        "/cluster/projects/uasc/Datasets/Blitvenica/dataset",
        "/cluster/projects/uasc/Datasets/Fram Strait 2017-18/dataset",
        "/cluster/projects/uasc/Datasets/Atwain 2017-18/dataset",
        "/cluster/projects/uasc/Datasets/Losinj/dataset",
    ]
    data_dirs = ["C:\\Users\\ander\\Github\\Masters\\Pdata"]
    data_module = DataloaderModule(data_dirs=data_dirs, batch_size=batch_size)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Load and print some batches
    for batch_idx, spectrograms in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}, Spectrograms Shape: {spectrograms.shape}")
        sys.stdout.flush()


if __name__ == "__main__":
    test_dataloader(batch_size=3)
