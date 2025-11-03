import math
import torch
import torchaudio
import os
import pretrain_dataloader

#AudioMAE baseline
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

    win_len_frames = int(round(frame_length * 120000 / 1_000.0))
    hop_len_frames = int(round(frame_shift * 120000 / 1_000.0))
    time_per_slize = ((256 - 1) * hop_len_frames + win_len_frames) / 120000
    print(f"Time per slize of {256} is {time_per_slize} s of audio")


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

# My own linear spectrogram
class FbankProcessor:
    """
    Computes a **log-power linear spectrogram** with window & hop
    specified in milliseconds.
    """
    def __init__(
            self,
            win_len_ms = 25,  # ms, ≈4096 @120 kHz
            hop_len_ms  = 10,  # ms, ≈2048 @120 kHz
            max_sample_rate  = 110_000,
    ):
        self.win_len_ms = win_len_ms
        self.hop_len_ms = hop_len_ms
        self.max_sample_rate = max_sample_rate
        self.power_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=100
        )

    @staticmethod
    def _ms_to_samples(ms: float, sr: int) -> int:
        return int(round(ms * sr / 1_000.0))

    def _build_spectrogram(self, sr: int):
        win_len = self._ms_to_samples(self.win_len_ms, sr)
        hop_len = self._ms_to_samples(self.hop_len_ms, sr)
        n_fft = 2 ** math.ceil(math.log2(win_len))  # next power of 2

        n_fft = win_len = 254
        hop_len = int(win_len // 2)
        print(f"win_len: {win_len}")
        print(f"hop_len: {hop_len}")
        print(f"overlap percentage {hop_len / win_len}")

        return torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            window_fn=torch.hann_window,
            center=False,
            power=2.0,
        ), win_len, hop_len

    def __call__(self, waveform: torch.Tensor, sample_rate: int):
        waveform = waveform - waveform.mean()

        if sample_rate > self.max_sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.max_sample_rate
            )(waveform)
            sample_rate = self.max_sample_rate

        waveform = torchaudio.functional.highpass_biquad(waveform = waveform, sample_rate = sample_rate, cutoff_freq = 20, Q = 0.707)

        spec_fn, win_len, hop_len = self._build_spectrogram(sample_rate)
        spectrogram = spec_fn(waveform)
        spectrogram = self.power_to_db(spectrogram)
        spectrogram = spectrogram.squeeze(0)

        # 0-1 normalisation (preserves dynamic range fractions)
        spectrogram =(spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        return spectrogram, sample_rate, win_len, hop_len


def prosess_file_modern(path, frame_offset=0, num_frames=-1):
    n_freq_bins = 128
    #n_fft = 4096
    #win_len = 3000#4096
    #hop_len = 1200#win_len // 4  # 50% overlap
    #win_len_ms = 15#15#34.13
    #hop_len_ms = 9#6#17.07


    frames_per_segment = 1024

    fbank_processor = FbankProcessor()

    waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    fbank, sr, win_len, hop_len = fbank_processor(waveform=waveform, sample_rate=sr)

    time_per_slize = ((frames_per_segment - 1) * hop_len + win_len) / sr
    print(f"Time per slize of {frames_per_segment} is {time_per_slize} s of audio")

    pretrain_dataloader.show_np_spectrogram_file(fbank.T, file_name=f"", vmax=1)

    slices = fbank.unfold(dimension=1, size=1024, step=512)
    slices = torch.permute(slices, (1, 2, 0))

    i = 0
    for slice in slices:
        pretrain_dataloader.show_np_spectrogram_file(slice,
                                                     file_name=f"",vmax=1)  # =path.split("\\")[-1],vmax=1, title="Modern")
        if i > 6:
            break
        i +=1


def prosess_file_like_audiomae(path, frame_offset=0, num_frames=-1):
    waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    fbank = convert_numpy_to_fbank(waveform, sr)
    fbank = fbank.transpose(0, 1)
    slices = fbank.unfold(dimension=1, size=256, step=128)
    slices = torch.permute(slices, (1, 2, 0))
    pretrain_dataloader.show_np_spectrogram_file(fbank.T,
                                                 file_name=f"")

    fbank2 = convert_numpy_to_fbank(waveform, sr, max_sample_rate=16000)#Base audioMAE
    pretrain_dataloader.show_np_spectrogram_file(fbank2,
                                                 file_name=f"")

    i = 0
    for slice in slices:
        pretrain_dataloader.show_np_spectrogram_file(slice,
                                                     file_name="")  # file_name=path.split("\\")[-1], title="AudioMAE")
        if i > 24:
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
    paths = [  #"C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\CMHS_2020_08_19_11_34_01.wav",
        #"C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\CMHS_2022_05_09_12_17_31.wav",
        #"C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\CMHS_2022_05_09_12_22_37.wav", # Sperm whale 10kHz
        #"C:\\Users\\ander\\OneDrive\\Masters\\Pdata\\2019_07_04_10_53_34.wav",  # Dolphin 69kHz
        "C:\\Users\\ander\\Github\\Masters\\Pdata\\Thesis Examples\\DolphClickandSong.wav",
        #"C:\\Users\\ander\\Github\\Masters\\Pdata\\Thesis Examples\\SpermSongNoise.wav",
         "C:\\Users\\ander\\Github\\Masters\\Pdata\\Thesis Examples\\SpermWhaleClick.wav",
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
