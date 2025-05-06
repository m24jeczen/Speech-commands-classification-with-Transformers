import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.vaggish import VGGishClassifier
from scipy.io import wavfile
import os

BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 0.001
#MODEL_SAVE_PATH = "vggish_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets & Dataloaders
DATA_DIR = "data/train/audio"
N = 5000
commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
print(DEVICE)

class SpeechCommandsDatasetRaw(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_index = {label: i for i, label in enumerate(commands)}

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

        if waveform.size(-1) < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(-1)))
        waveform = waveform[:, :16000]

        label_idx = self.label_to_index[label]
        return waveform, label_idx

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

combined = list(zip(all_files, all_labels))
random.shuffle(combined)
all_files, all_labels = zip(*combined)

# Train/val/test split
train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.3, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)

# Dataset and Dataloaders
train_dataset = SpeechCommandsDatasetRaw(train_files, train_labels)
val_dataset = SpeechCommandsDatasetRaw(val_files, val_labels)
test_dataset = SpeechCommandsDatasetRaw(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
model = VGGishClassifier(num_classes=len(commands)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in tqdm(range(EPOCHS)):
    model.train()
    running_loss, correct = 0.0, 0

    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_dataset)
    print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}, Train Accuracy = {train_acc:.4f}")

print("Training complete.")

model.eval()
correct = 0
print("Validating...")
with torch.no_grad():
    for waveforms, labels in val_loader:
        waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)
        outputs = model(waveforms)
        correct += (outputs.argmax(1) == labels).sum().item()

val_acc = correct / len(val_dataset)
print(f"Validation Accuracy = {val_acc:.4f}")

