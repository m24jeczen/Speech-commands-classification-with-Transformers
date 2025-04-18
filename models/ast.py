from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch
import torch.nn as nn
import torch.optim as optim

class ASTModelWrapper:
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=12):
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # ← this fixes the error
        )

    def preprocess(self, waveform, sample_rate):
        return self.feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")

def train_model(model_name, train_loader, val_loader, num_labels=12, epochs=10, lr=1e-4):
    if model_name != "ast":
        raise NotImplementedError("Only AST model is implemented in this module.")
    
    ast = ASTModelWrapper(num_labels=num_labels)
    model = ast.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

    return model
