import os
import torch
from torch.utils.data import DataLoader
from config import AUDIO_DIR, LABELS, SAMPLE_RATE
from utils.dataset import SpeechDataset
from models.model_selector import get_model
from train import train_model
from utils.audio_utils import create_mel_spectrogram
from transformers import ASTFeatureExtractor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

# 🔇 Disable TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_ast_collate_fn(feature_extractor):
    def collate_fn(batch):
        input_values = []
        labels = []

        for waveform, label in batch:
            if waveform.shape[0] < 400:
                waveform = pad(waveform, (0, 400 - waveform.shape[0]))

            waveform = waveform.numpy()
            inputs = feature_extractor(
                waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )["input_values"][0]
            input_values.append(inputs)
            labels.append(label)

        padded_inputs = pad_sequence(input_values, batch_first=True)
        return {"input_values": padded_inputs}, torch.tensor(labels)

    return collate_fn

from transformers import ASTFeatureExtractor
from utils.collate import ASTCollateFn
def main():
    torch.manual_seed(42)

    # Sanity check: mel spec preview
    example_path = f"{AUDIO_DIR}/yes/0a7c2a8d_nohash_0.wav"
    mel = create_mel_spectrogram(example_path)
    print(f"Mel Spectrogram Shape: {mel.shape}")

    # 🧠 Choose model
    model_name = "ast"  # or 'cnn', 'mlp'

    # 📦 Load dataset
    dataset = SpeechDataset(AUDIO_DIR, LABELS, mode=model_name)

    # 🧱 Collate function
    if model_name == "ast":
        feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        collate_fn = ASTCollateFn(feature_extractor)  # 👈 Now a class, not a closure!
    else:
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate

    # 🚚 DataLoader
    dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,  # ✅ works now!
    pin_memory=True,
    persistent_workers=True
    )

    # 🧠 Get model
    model = get_model(model_name=model_name, num_labels=len(LABELS))

    # 🏋️ Train
    trained_model = train_model(
        model=model,
        model_name=model_name,
        train_loader=dataloader,
        val_loader=None,
        epochs=3,
        lr=1e-4
    )

    print("✅ Training completed!")


if __name__ == "__main__":
    main()
