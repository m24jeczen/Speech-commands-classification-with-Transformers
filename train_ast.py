import os
import torch
import torchaudio
import csv
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = 'data/train/audio'
SAMPLE_RATE = 16000
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
PATCH_SIZE = 16  
DEBUG_MODE = True
SAVE_MODEL_PATH = "ast_classifier.pt"
IS_BACKGROUND_NOISE_USED = False
RESULTS_FILE = "test_results.txt"
REPEAT_DEBUG_RUNS = 5

if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        f.write("Experiment Results\n")

class ASTClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(ASTClassifier, self).__init__()
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, input_values):
        return self.model(input_values).logits

class SpeechCommandsDataset(Dataset):
    def __init__(self, filepaths, labels, extractor):
        self.filepaths = filepaths
        self.labels = labels
        self.extractor = extractor

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        waveform, sr = torchaudio.load(os.path.normpath(filepath))
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(dim=0)
        input_values = self.extractor(waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.squeeze(0)
        return input_values, self.labels[idx]

def load_dataset():
    core_commands = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    filepaths, labels = [], []

    for label in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(dir_path) or label == "_background_noise_":
            continue

        actual_label = label if label in core_commands else "unknown"

        for wav in os.listdir(dir_path):
            if wav.endswith(".wav"):
                filepaths.append(os.path.join(dir_path, wav))
                labels.append(actual_label)

    return filepaths, labels

def split_dataset(filepaths, labels):
    train_files, test_files, train_labels, test_labels = train_test_split(filepaths, labels, test_size=0.3, stratify=labels)
    return train_files, test_files, train_labels, test_labels

def plot_confusion_matrix(y_true, y_pred, labels, prefix=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{prefix}_confusion_matrix_200.png")
    plt.close()

def save_results_to_file(prefix, avg_loss, avg_acc, avg_f1):
    result_line = f"{prefix} 200| loss: {avg_loss:.4f}, acc: {avg_acc:.2f}%, f1: {avg_f1:.4f}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(result_line)

def run_multiple_debug_experiments():
    all_f1, all_acc, all_loss = [], [], []

    for run in range(1, REPEAT_DEBUG_RUNS + 1):
        logging.info(f"--- DEBUG RUN {run} ---")
        loss, acc, f1 = train(EXPERIMENT_NUMBER=run)
        all_loss.append(loss)
        all_acc.append(acc)
        all_f1.append(f1)

    prefix = f"{NUM_EPOCHS}ep_{LEARNING_RATE}lr_{BATCH_SIZE}bs_{'yes' if IS_BACKGROUND_NOISE_USED else 'no'}bg_debug_avg_200"
    save_results_to_file(prefix, np.mean(all_loss), np.mean(all_acc), np.mean(all_f1))

def initialize_csv_logger(num_epochs, lr, batch_size, use_background_noise, experiment_number):
    filename = f"train_log_{num_epochs}ep_{lr}lr_{batch_size}bs_{'yes' if use_background_noise else 'no'}bg_exp_{experiment_number}_200.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "accuracy", "f1"])
    return filename

def append_to_csv_logger(filename, epoch, loss, acc, f1):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, round(loss, 4), round(acc, 2), round(f1, 4)])

def train(EXPERIMENT_NUMBER=1):
    logging.info("Loading data...")
    filepaths, raw_labels = load_dataset()

    df = pd.DataFrame({"file": filepaths, "label": raw_labels})
    if DEBUG_MODE:
        df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), 200)))
    filepaths = df["file"].tolist()
    raw_labels = df["label"].tolist()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(raw_labels)
    NUM_CLASSES = len(label_encoder.classes_)
    logging.info(f"Classes: {label_encoder.classes_}")

    train_files, test_files, train_labels, test_labels = split_dataset(filepaths, encoded_labels)
    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    train_dataset = SpeechCommandsDataset(train_files, train_labels, extractor)
    test_dataset = SpeechCommandsDataset(test_files, test_labels, extractor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0
    train_log_file = initialize_csv_logger(NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, IS_BACKGROUND_NOISE_USED, EXPERIMENT_NUMBER)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(train_loader)

        model.eval()
        all_train_preds, all_train_targets = [], []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_targets.extend(labels.cpu().numpy())

        train_acc = np.mean(np.array(all_train_preds) == np.array(all_train_targets)) * 100
        train_f1 = f1_score(all_train_targets, all_train_preds, average="macro")
        append_to_csv_logger(train_log_file, epoch+1, mean_loss, train_acc, train_f1)
        logging.info(f"Epoch {epoch+1} Loss: {mean_loss:.4f} - Train Accuracy: {train_acc:.2f}% - Train F1: {train_f1:.4f}")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    f1 = f1_score(all_targets, all_preds, average="macro")
    logging.info(f"Test Accuracy: {acc:.2f}% - F1 Score: {f1:.4f}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    logging.info(f"Model saved with F1: {f1:.4f}")
    plot_confusion_matrix(
        all_targets,
        all_preds,
        label_encoder.classes_,
        prefix=f"{NUM_EPOCHS}ep_{LEARNING_RATE}lr_{BATCH_SIZE}bs_{'yes' if IS_BACKGROUND_NOISE_USED else 'no'}bg_exp_{EXPERIMENT_NUMBER}"
    )
    logging.info("Confusion matrix saved.")
    logging.info("Test Classification report:\n" + classification_report(all_targets, all_preds, target_names=label_encoder.classes_))
    save_results_to_file(f"{NUM_EPOCHS}ep_{LEARNING_RATE}lr_{BATCH_SIZE}bs_{'yes' if IS_BACKGROUND_NOISE_USED else 'no'}bg_exp_{EXPERIMENT_NUMBER}", mean_loss, acc, f1)

    return mean_loss, acc, f1

if __name__ == "__main__":
    if DEBUG_MODE:
        run_multiple_debug_experiments()
    else:
        train()