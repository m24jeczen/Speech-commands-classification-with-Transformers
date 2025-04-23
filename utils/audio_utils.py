# utils/audio_utils.py or a new file called mel_spectrogram.py
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
