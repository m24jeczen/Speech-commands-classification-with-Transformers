import torchaudio
import torchaudio.transforms as T

def create_mel_spectrogram(filepath, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
    waveform, sr = torchaudio.load(filepath)
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        waveform = resample(waveform)

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)

    mel_db = T.AmplitudeToDB(top_db=80)(mel_spectrogram)
    return mel_db



# import os
# import glob
# import librosa
# import numpy as np
# import librosa.display
# import matplotlib.pyplot as plt
# from datetime import datetime

# def load_audio_data(data_dir='.//data', sr=16000, duration=1.0, is_train=True):
#     """
#     Load audio files from the train or test directory.

#     Args:
#         data_dir (str): Path to the data directory containing 'train' or 'test' subdirectories.
#         sr (int): Sampling rate for audio files.
#         duration (float): Desired duration (in seconds) for each audio file.
#         is_train (bool): If True, loads from 'train' folder and returns labels. Else, loads from 'test'.

#     Returns:
#         list of (waveform, label) tuples if is_train else list of waveforms
#     """
#     folder = 'train' if is_train else 'test'
#     path = os.path.join(data_dir, folder)

#     audio_data = []
#     labels = []

#     if is_train:
#         for label_dir in os.listdir(path):
#             label_path = os.path.join(path, label_dir)
#             if os.path.isdir(label_path):
#                 for file in glob.glob(os.path.join(label_path, '*.wav')):
#                     waveform, _ = librosa.load(file, sr=sr, duration=duration)
#                     audio_data.append(waveform)
#                     labels.append(label_dir)
#         return list(zip(audio_data, labels))
#     else:
#         for file in glob.glob(os.path.join(path, '*.wav')):
#             waveform, _ = librosa.load(file, sr=sr, duration=duration)
#             audio_data.append(waveform)
#         return audio_data

# import os
# import glob
# import torch
# from torch.utils.data import Dataset
# import librosa
# import numpy as np

# class SpeechCommandsDataset(Dataset):
#     def __init__(self, data_dir='data', sr=16000, n_mels=128, duration=1.0,
#                  is_train=True, label_map=None):
#         """
#         Args:
#             data_dir (str): Base directory with 'train' and 'test' folders.
#             sr (int): Sampling rate.
#             n_mels (int): MEL bands.
#             duration (float): Duration in seconds to trim/pad audio.
#             is_train (bool): Load train or test data.
#             label_map (dict): Optional label-to-index mapping.
#         """
#         self.sr = sr
#         self.n_mels = n_mels
#         self.n_samples = int(sr * duration)
#         self.is_train = is_train
#         self.files = []
#         self.labels = []

#         folder = 'train' if is_train else 'test'
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#         base_path = os.path.join(project_root, data_dir, folder, 'audio')

#         if is_train:
#             for label in sorted(os.listdir(base_path)):
#                 label_path = os.path.join(base_path, label)
#                 if os.path.isdir(label_path):
#                     for file in glob.glob(os.path.join(label_path, '*.wav')):
#                         self.files.append(file)
#                         self.labels.append(label)
#             self.label_map = label_map or {lbl: idx for idx, lbl in enumerate(sorted(set(self.labels)))}
#         else:
#             self.files = glob.glob(os.path.join(base_path, '*.wav'))
#             self.label_map = {}

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         filepath = self.files[idx]
#         waveform, _ = librosa.load(filepath, sr=self.sr)
#         # Pad or truncate to fixed length
#         if len(waveform) < self.n_samples:
#             waveform = np.pad(waveform, (0, self.n_samples - len(waveform)))
#         else:
#             waveform = waveform[:self.n_samples]

#         # MEL spectrogram
#         mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sr, n_mels=self.n_mels)
#         mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#         mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32)

#         if self.is_train:
#             label = self.labels[idx]
#             label_idx = self.label_map[label]
#             return mel_tensor.unsqueeze(0), label_idx  # Add channel dim for CNN
#         else:
#             return mel_tensor.unsqueeze(0)  # No label for test data

# def waveform_to_mel(waveform, sr=16000, n_mels=128, n_fft=1024, hop_length=512,
#                     show=False, save=False, save_dir='spectrograms', filename=None):
#     """
#     Convert a waveform to a MEL spectrogram, and optionally show/save as an image.

#     Args:
#         waveform (np.array): Audio waveform.
#         sr (int): Sample rate.
#         n_mels (int): Number of MEL bands.
#         n_fft (int): FFT window size.
#         hop_length (int): Number of samples between frames.
#         show (bool): Whether to display the spectrogram with matplotlib.
#         save (bool): Whether to save the image to disk.
#         save_dir (str): Folder where image will be saved (default: 'spectrograms').
#         filename (str): Optional filename for saving (e.g. 'yes_01.png').

#     Returns:
#         np.ndarray: MEL spectrogram (dB-scaled).
#     """
#     # Generate MEL spectrogram
#     mel_spec = librosa.feature.melspectrogram(
#         y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
#     )
#     mel_db = librosa.power_to_db(mel_spec, ref=np.max)

#     # Visualization
#     if show or save:
#         plt.figure(figsize=(8, 4))
#         librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length,
#                                  x_axis='time', y_axis='mel', cmap='magma')
#         plt.colorbar(format='%+2.0f dB')
#         plt.title('MEL Spectrogram')
#         plt.tight_layout()

#         if save:
#             os.makedirs(save_dir, exist_ok=True)
#             if not filename:
#                 timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#                 filename = f"mel_spec_{timestamp}.png"
#             save_path = os.path.join(save_dir, filename)
#             plt.savefig(save_path)
#             print(f"✅ Spectrogram saved to: {save_path}")

#         if show:
#             plt.show()

#         plt.close()

#     return mel_db

# class ASTDataset(Dataset):
#     def __init__(self, data_dir='data', is_train=True, duration=1.0, sr=16000):
#         self.sr = sr
#         self.n_samples = int(sr * duration)
#         self.files = []
#         self.labels = []

#         folder = 'train' if is_train else 'test'
#         base_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), data_dir, folder, 'audio')

#         for label in sorted(os.listdir(base_path)):
#             label_path = os.path.join(base_path, label)
#             if os.path.isdir(label_path):
#                 for f in glob.glob(os.path.join(label_path, '*.wav')):
#                     self.files.append(f)
#                     self.labels.append(label)

#         self.label_to_idx = {lbl: i for i, lbl in enumerate(sorted(set(self.labels)))}

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         label = self.labels[idx]

#         waveform, _ = librosa.load(path, sr=self.sr)
#         if len(waveform) < self.n_samples:
#             waveform = torch.nn.functional.pad(torch.tensor(waveform), (0, self.n_samples - len(waveform)))
#         else:
#             waveform = torch.tensor(waveform[:self.n_samples])

#         return waveform, self.label_to_idx[label]
    

# import torchaudio
# import torchaudio.transforms as T

# def wav_to_mel(file_path, sample_rate=16000, n_mels=128):
#     waveform, sr = torchaudio.load(file_path)
#     if sr != sample_rate:
#         resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
#         waveform = resampler(waveform)
#     mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
#     mel_spectrogram = torchaudio.functional.amplitude_to_DB(mel_spectrogram, multiplier=10.0, amin=1e-10, db_multiplier=0)
#     return mel_spectrogram
