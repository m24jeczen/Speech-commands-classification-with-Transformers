import torchaudio
import torchaudio.transforms as T

def get_mel_spectrogram(waveform, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)
    mel_spectrogram = torchaudio.functional.amplitude_to_DB(mel_spectrogram, multiplier=10.0, db_multiplier=0.0, amin=1e-10, top_db=80.0)
    return mel_spectrogram


import os
import wave
import numpy as np
from scipy.io import wavfile

def generate_silence_clips(background_path, output_path, stride_sec=0.25, clip_duration_sec=1.0):
    os.makedirs(output_path, exist_ok=True)
    sample_rate = 16000 

    clip_count = 0

    for filename in os.listdir(background_path):
        if filename.endswith('.wav'):
            filepath = os.path.join(background_path, filename)
            rate, data = wavfile.read(filepath)

            if data.ndim > 1:
                data = data[:, 0]  

            total_samples = len(data)
            clip_len_samples = int(clip_duration_sec * rate)
            stride_samples = int(stride_sec * rate)

            for start in range(0, total_samples - clip_len_samples + 1, stride_samples):
                end = start + clip_len_samples
                clip = data[start:end]

                out_filename = f"silence_{clip_count:05d}.wav"
                wavfile.write(os.path.join(output_path, out_filename), rate, clip.astype(np.int16))
                clip_count += 1

    print(f"Generated {clip_count} 1-sec fragments into: {output_path}")