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
