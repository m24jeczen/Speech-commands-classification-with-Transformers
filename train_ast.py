# train_ast.py
import os
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor
from models.ast import ASTClassifier
from utils.audio_utils import get_mel_spectrogram
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np

DATA_DIR = "data/train/audio"
SAMPLE_RATE = 16000
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_CLASSES = 12  # Adjust this based on your final label count

class SpeechCommandsDataset(Dataset):
    def __init__(self, filepaths, labels, extractor):
        self.filepaths = filepaths
        self.labels = labels
        self.extractor = extractor

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        waveform, sr = torchaudio.load(filepath)
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(dim=0)  # convert to mono if not already
        input_values = self.extractor(
            waveform.numpy(),  # no spectrogram, just raw waveform
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_values.squeeze(0)
        label = self.labels[idx]
        return input_values, label

def load_dataset():
    core_commands = [
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
    ]
    all_dirs = os.listdir(DATA_DIR)

    filepaths = []
    labels = []

    for label in all_dirs:
        full_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(full_path):
            continue

        actual_label = label
        if label == "_background_noise_":
            continue  # silence is handled separately if needed
        elif label not in core_commands:
            actual_label = "unknown"  # collapse all other classes to 'unknown'

        for wav in os.listdir(full_path):
            if wav.endswith(".wav"):
                filepaths.append(os.path.join(full_path, wav))
                labels.append(actual_label)

    return filepaths, labels

def train():
    print("Loading data...")
    filepaths, raw_labels = load_dataset()

    # 💡 Only use a small subset for debugging
    DEBUG_SIZE = 500
    df = pd.DataFrame({"file": filepaths, "label": raw_labels})
    df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), 40)))  # 40 samples per class (adjustable)
    filepaths = df["file"].tolist()
    raw_labels = df["label"].tolist()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(raw_labels)
    NUM_CLASSES = len(label_encoder.classes_)
    print("Classes:", label_encoder.classes_)

    train_files, val_files, train_labels, val_labels = train_test_split(filepaths, encoded_labels, test_size=0.2, stratify=encoded_labels)

    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    train_dataset = SpeechCommandsDataset(train_files, train_labels, extractor)
    val_dataset = SpeechCommandsDataset(val_files, val_labels, extractor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Training started...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # Validation loop (optional)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
