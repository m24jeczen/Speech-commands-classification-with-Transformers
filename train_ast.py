import torch
from torch.utils.data import DataLoader
from models.ast_classifier import ASTClassifier
from utils.dataset_loader import ASTDataset
from transformers import get_scheduler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup
model = ASTClassifier(num_labels=35).to(device)
dataset = ASTDataset(data_dir='data', is_train=True)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 3
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for waveforms, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs = model.extract_features(waveforms.numpy(), sampling_rate=16000)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    
    print(f"✅ Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")
