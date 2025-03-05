import json
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import matplotlib
import matplotlib.pyplot as plt
import os
from datetime import datetime

#from IPython.core.pylabtools import figsize


def convert_and_slice_wav_to_mel_spectrogram_x_sec(input_folder: str, output_folder: str, n_fft=1024, hop_length=None, n_mels=128, target_length_seconds=10):
    hop_length = hop_length if hop_length else n_fft // 4
    parameters = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "target_length_seconds": target_length_seconds,
    }
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_folder, file_name)
            samples, sample_rate = librosa.load(file_path, sr=None)

            total_duration = len(samples) / sample_rate
            num_segments = int(total_duration // target_length_seconds)

            for i in range(num_segments):
                start_time = i * target_length_seconds
                end_time = (i + 1) * target_length_seconds
                segment_samples = samples[int(start_time * sample_rate):int(end_time * sample_rate)]

                # Convert the segment to a Mel spectrogram
                mel_spectrogram_db = convert_numpy_to_fbank(segment_samples, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, target_length_seconds=target_length_seconds-2)
                #mel_spectrogram_db = slice_and_pad_segment(mel_spectrogram_db, target_length, 3 / 4, 0.3)

                # Save the spectrogram as a numpy array
                output_file_path = os.path.join(output_folder, f"{file_name.replace('.wav', '')}_segment_{i}_QRY.npy")
                np.save(output_file_path, mel_spectrogram_db)

    # Save parameters used for spectrogram generation
    with open(os.path.join(output_folder, "parameters.json"), "w") as param_file:
        json.dump(parameters, param_file, indent=4)

def convert_and_slice_wav_to_kaldi_spectrogram(input_folder: str, output_folder: str, csv_path: str, n_fft=1024, hop_length=None, n_mels=128, target_length_seconds=10):
    hop_length = hop_length if hop_length else n_fft // 4
    parameters = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels
    }
    os.makedirs(output_folder, exist_ok=True)
    csv_data = pd.read_csv(csv_path)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_folder, file_name)
            samples, sample_rate = librosa.load(file_path, sr=None)
            base_name = os.path.basename(file_name)
            start_datetime_str = base_name.split('_')[0] + '_' + base_name.split('_')[1]
            start_datetime = datetime.strptime(start_datetime_str, "%y%m%d_%H%M%S")

            # Get all annotations for this file
            name_in_file = "E:\\Atwain_2017-18\\Renamed data\\2017-18\\ATW_dec\\" + file_name.replace(".wav", ".d32.wav")
            annotations = csv_data[csv_data['Input file'] == name_in_file]
            total_audio_length = len(samples) / sample_rate
            segments = []
            previous_end_time = 0
            for _, row in annotations.iterrows():
                call_start_time = datetime.strptime(row['Start time'], "%m/%d/%Y %H:%M:%S.%f")
                call_end_time = datetime.strptime(row['End time'], "%m/%d/%Y %H:%M:%S.%f")
                offset_start = (call_start_time - start_datetime).total_seconds()
                offset_end = (call_end_time - start_datetime).total_seconds()

                # Negative sample before the positive one
                if previous_end_time < offset_start:
                    adjusted_start = previous_end_time
                    adjusted_end = max(offset_start, adjusted_start + target_length_seconds)
                    if adjusted_end <= total_audio_length:
                        segments.append((adjusted_start, adjusted_end, 'NON'))

                # Positive call sample
                adjusted_start = offset_start
                adjusted_end = max(offset_end, adjusted_start + target_length_seconds)
                if adjusted_end <= total_audio_length:
                    segments.append((adjusted_start, adjusted_end, row['Species Code']))

                previous_end_time = offset_end

            for start, end, label in segments:
                segment_samples = samples[int(start * sample_rate):int(end * sample_rate)]
                mel_spectrogram_db = convert_numpy_to_fbank(segment_samples, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, target_length_seconds=target_length_seconds-2)
                output_file_path = os.path.join(output_folder, f"{file_name.replace('.wav', '')}_{start}_{end}_{label}.npy")
                np.save(output_file_path, mel_spectrogram_db)

    #save param
    with open(os.path.join(output_folder, "parameters.json"), "w") as param_file:
        json.dump(parameters, param_file, indent=4)

def convert_numpy_to_fbank(numpy_array, sample_rate, n_mels, n_fft, hop_length, target_length_seconds):
    """
    Converts a numpy array (audio segment) to a mel spectrogram using torchaudio's fbank function.
    """
    waveform = torch.from_numpy(numpy_array.copy()).float() # Weird error when i do not create a copy of the tensor
    waveform = waveform - waveform.mean()
    waveform = waveform.unsqueeze(0)
    high_freq = 100
    frame_length = 8192/8 # (32768 // sample_rate) * 1000
    frame_shift = frame_length // 4
    number_of_frames = int(target_length_seconds / (frame_shift/1000))
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=128, #n_mels,
        low_freq=0.0,
        high_freq = high_freq,
        dither=0.0,
        frame_shift=frame_shift,
        frame_length= frame_length,
    )

    n_frames = fbank.shape[0]
    p = number_of_frames - n_frames
    #if p > 0:
    #    # Pad with zeros if less than the target length
    #    m = torch.nn.ZeroPad2d((0, 0, 0, p))
    #    fbank = m(fbank)
    if p < 0:
        # Cut if more than the target length
        fbank = fbank[0:number_of_frames, :]
    return fbank


def extract_filenames_from_csv(csv_path: str):
    csv_data = pd.read_csv(csv_path)
    filenames = csv_data['Input file'].apply(lambda x: os.path.basename(x).replace('.d32', '')).unique()
    return filenames.tolist()


def split_filenames(filenames, max_length=255):
    chunks = []
    current_chunk = []
    current_length = 0

    for filename in filenames:
        length = len(filename) + 3  # Including extra characters for comma and braces
        if current_length + length > max_length:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0

        current_chunk.append(filename)
        current_length += length

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_copy_commands(chunks, source_folder, destination_folder):
    commands = []
    for chunk in chunks:
        filenames_str = "{" + ",".join(chunk) + "}"
        command = f"cp {source_folder}/{filenames_str} {destination_folder}"
        commands.append(command)

    return commands

def show_np_spectrograms(folder):
    i = 0
    for file_name in os.listdir(folder):
        if file_name.endswith(".npy"):
            spec = np.load(folder + "\\" + file_name)
            spec = spec.T #Transpose for better view
            print(spec.shape)
            #plt.figure(figsize=(25, 5))
            img = plt.imshow(20 * spec, cmap='turbo', interpolation='none', origin='lower', aspect='equal')
            plt.ylim(0, spec.shape[0])
            plt.xlim(0, spec.shape[1])
            plt.xlabel('Samples')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(img)
            plt.title(file_name)
            plt.show()
            #plt.savefig(f"temp\\{file_name}.png", format='png')
            i+=1
            #if i > 8: break

def main():
    # Example usage to convert wav files to mel spectrograms
    csv_path = "Atwain_2017_log_detections.csv"
    input_wav_folder = "/cluster/projects/uasc/anders/datasets/PData"
    output_spectrogram_folder = "/cluster/projects/uasc/anders/datasets/dataset_trunct"
    convert_and_slice_wav_to_kaldi_spectrogram(input_folder=input_wav_folder, output_folder=output_spectrogram_folder, csv_path=csv_path, target_length_seconds=12)
    show_np_spectrograms(output_spectrogram_folder)

    #input_wav_folder = "/cluster/projects/uasc/Datasets/Fram Strait data/Atwain_2017-18/Renamed data/2017-18/"
    #output_spectrogram_folder = "/cluster/projects/uasc/anders/datasets/dataset_query_trunct"
    #convert_and_slice_wav_to_mel_spectrogram_x_sec(input_folder=input_wav_folder, output_folder=output_spectrogram_folder, target_length_seconds=12)
    #show_np_spectrograms(output_spectrogram_folder)

    #filenames = extract_filenames_from_csv(os.path.join(input_wav_folder, "Atwain_2017_log_detections.csv"))
    #chunks = split_filenames(filenames)
    #commands = generate_copy_commands(chunks, "/cluster/projects/uasc/Datasets/'Fram Strait data'/Atwain_2017-18/'Renamed data'/2017-18", "/cluster/home/andernh/PData")
    #for command in commands:
    #    print(command)
    #sample_rate, samples = wavfile.read('Data/180318_020000_AU_AT04.wav')
    #show_spectrogram(sample_rate, samples)
    #show_mel_spectrogram(sample_rate, samples)
    #show_frequency_distribution(sample_rate, samples)

if __name__ == '__main__':
    main()
