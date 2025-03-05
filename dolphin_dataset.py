import os

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import util
from tqdm import tqdm


class DolphinDatasetConstructor:
    labels = None

    def __init__(self, label_path):
        df = pd.read_excel(label_path)
        df["Path"] = df["Path"].str.strip()
        print(df.info())
        self.labels = dict(zip(df["Path"], df["Species"]))
        print(self.labels)

    def convert_numpy_to_fbank(self, numpy_array, sample_rate):
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
            self,
            output_folder_path,
            file_path,
            fbank,
            sample_rate,
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

        :param output_folder_path: Where to save .npy slices
        :param file_path: Original file path (used to derive output filename)
        :param fbank: Full spectrogram, shape [num_frames, n_mels]
        :param sample_rate: Audio sample rate (only used if needed for reference)
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

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        start_frame = 0
        while (start_frame + frames_per_segment) <= fbank.shape[0]:
            end_frame = start_frame + frames_per_segment
            segment = fbank[start_frame:end_frame, :]  # shape: [frames_per_segment, n_mels]

            #file_path = "/cluster/projects/uasc/Datasets/Data univ Bari/dataset/" + file_path.split("\\")[-1].replace("CMHS_", "").replace("TARANTO_","") # For testing on my own computer, artificially adjusting path
            #file_path = file_path.replace("Data univ Bari", "Data univ Bari copy")  # There is a naming error in the file compared to the original files
            label = ""
            try:
                label = self.labels[file_path]
            except KeyError:
                pass

            output_file_name = f"{base_name}_{start_frame}_{end_frame}_{label}.npy"
            output_file_path = os.path.join(output_folder_path, output_file_name)

            np.save(output_file_path, segment.numpy())
            start_frame += step_size

    def process_audio_files(
            self,
            folder_path,
            output_folder_path,
            overlap_percentage=0.0,
            target_length_seconds=20.0,
    ):
        """
        1. Iterates over each .wav file in 'folder_path'.
        2. Loads the entire audio.
        3. Converts to full FBank spectrogram (no padding / no slicing).
        4. Slices the spectrogram in time.
        5. Saves each slice as .npy in 'output_folder_path'.
        """
        num_files = len([name for name in os.listdir(folder_path)])
        print(f"Converting {num_files} to spectrograms")

        os.makedirs(folder_path, exist_ok=True)

        p_bar = tqdm(range(num_files))
        i = 0
        for file_name in os.listdir(folder_path):

            p_bar.update(1)
            p_bar.refresh()
            file_path = os.path.join(folder_path, file_name)

            if not os.path.isfile(file_path):
                continue
            if not file_name.lower().endswith('.wav'):
                continue

            # Load the entire audio
            audio_samples, sr = librosa.load(file_path, sr=None)

            # 1) Convert the entire audio to a full spectrogram
            full_fbank = self.convert_numpy_to_fbank(audio_samples, sr)

            # 2) Slice the spectrogram
            self.slice_spectrogram(
                output_folder_path=output_folder_path,
                file_path=file_path,
                fbank=full_fbank,
                sample_rate=sr,
                target_length_seconds=target_length_seconds,
                overlap_percentage=overlap_percentage,
                frame_shift_ms=10.0
            )
        p_bar.close()


def main():
    target_length_seconds = 10
    overlap_percentage = 0.5

    csv_folder = "/cluster/projects/uasc/Datasets/Data univ Bari/output.xlsx"
    input_wav_folder = "/cluster/projects/uasc/Datasets/Data univ Bari/dataset/"
    output_spectrogram_folder = "/cluster/projects/uasc/anders/datasets/dataset_trunct"

    #csv_folder = "C:\\Users\\ander\\GitHub\\Prosjekt\\output.xlsx"
    #input_wav_folder = "C:\\Users\\ander\\GitHub\\Prosjekt\\PData"
    #output_spectrogram_folder = "C:\\Users\\ander\\GitHub\\Prosjekt\\dataset_trunct"

    dolphin_dataset_constructor = DolphinDatasetConstructor(csv_folder)

    dolphin_dataset_constructor.process_audio_files(folder_path=input_wav_folder, output_folder_path=output_spectrogram_folder, target_length_seconds=target_length_seconds, overlap_percentage=overlap_percentage)
    util.show_np_spectrograms(output_spectrogram_folder)


# 4.52, 4.53, 4.57 for n_mels 128
# 4.53

if __name__ == '__main__':
    main()
