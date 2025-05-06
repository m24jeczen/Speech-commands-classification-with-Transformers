import os
import torch
import torchaudio
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking
from torchvision import transforms
from tqdm import tqdm
from scipy.io import wavfile
from models.cnn import CNN

# torchaudio.set_audio_backend("soundfile")
torchaudio.set_audio_backend("sox_io")
torch.manual_seed(42)

DATA_DIR = "data/train/audio"
N = 5000  # or however many you want total

commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

class SpeechCommandsDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_index = {label: i for i, label in enumerate(commands)}
        self.mel_spec = MelSpectrogram(sample_rate=16000, n_mels=128)
        self.db_transform = AmplitudeToDB()
        self.freq_mask = FrequencyMasking(freq_mask_param=15)
        self.time_mask = TimeMasking(time_mask_param=35)

    def load_waveform(self, path):
        sample_rate, data = wavfile.read(path)
        data = data.astype('float32') / 32768.0
        waveform = torch.tensor(data).unsqueeze(0)
        return waveform, sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        label = self.labels[index]
        waveform, sample_rate = self.load_waveform(path)

        mel = self.mel_spec(waveform)
        mel = self.db_transform(mel)
        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        mel = mel.squeeze(0).unsqueeze(0)

        if mel.size(-1) < 128:
            mel = torch.nn.functional.pad(mel, (0, 128 - mel.size(-1)))
        mel = mel[:, :, :128]

        label_idx = self.label_to_index[label]
        return mel, label_idx

all_files = []
all_labels = []

for label in commands:
    dir_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(dir_path):
        continue
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.wav')]
    selected = random.sample(files, min(len(files), N // len(commands)))  # even dist
    all_files.extend(selected)
    all_labels.extend([label] * len(selected))

# Shuffle before splitting
combined = list(zip(all_files, all_labels))
random.shuffle(combined)
all_files, all_labels = zip(*combined)

# Train/val/test split
train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.3, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)

# Dataset and Dataloaders
train_dataset = SpeechCommandsDataset(train_files, train_labels)
val_dataset = SpeechCommandsDataset(val_files, val_labels)
test_dataset = SpeechCommandsDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=len(commands)).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in tqdm(range(10)):
    model.train()
    total_loss, correct = 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {acc:.4f}")

# Save model
#torch.save(model.state_dict(), "audio_command_model.pth")
