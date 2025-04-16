from utils.dataset_loader import waveform_to_mel
from torch.utils.data import DataLoader
from utils.dataset_loader import SpeechCommandsDataset

# Train dataset
train_dataset = SpeechCommandsDataset(data_dir='data')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Peek at one batch
for mel, label in train_loader:
    print("Batch MEL shape:", mel.shape)
    print("Labels:", label)
    break

# Get a single sample from the batch for visualization
mel_batch, label_batch = next(iter(train_loader))
mel = mel_batch[0].squeeze().numpy()
label = label_batch[0].item()

# Plot the MEL spectrogram
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(8, 4))
librosa.display.specshow(mel, sr=16000, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title(f'MEL Spectrogram (Label ID: {label})')
plt.tight_layout()
plt.show()
