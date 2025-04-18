# predict.py

import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor
import pandas as pd
import os

from models.ast_classifier import ASTClassifier
from utils.dataset_loader import ASTDataset
import config

device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

# Load model
model = ASTClassifier(model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS).to(device)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.eval()

# Load test set (assume similar structure to train/audio/)
test_dataset = ASTDataset(data_dir=config.DATA_DIR, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Label mappings from training
label_map = test_dataset.label_to_idx
idx_to_label = {v: k for k, v in label_map.items()}

predictions = []
file_names = []

print("🔍 Running predictions on test set...")

with torch.no_grad():
    for waveform, _ in test_loader:
        waveform = waveform.squeeze(0)
        inputs = model.extract_features(waveform.numpy(), sampling_rate=config.SAMPLE_RATE)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        logits = model(inputs)
        predicted = torch.argmax(logits, dim=1).item()
        predictions.append(idx_to_label[predicted])
        file_names.append("file_" + str(len(file_names)))  # Add actual filenames if needed

# Save predictions
df = pd.DataFrame({'filename': file_names, 'prediction': predictions})
os.makedirs(os.path.dirname(config.PREDICTIONS_SAVE_PATH), exist_ok=True)
df.to_csv(config.PREDICTIONS_SAVE_PATH, index=False)
print(f"✅ Predictions saved to {config.PREDICTIONS_SAVE_PATH}")
