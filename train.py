# import torch
# from models.ast import ASTModelWrapper

# def train_model(model_name, train_loader, val_loader=None, num_labels=12, epochs=10, lr=1e-4):
#     if model_name == "cnn":
#         model = CNNModel(num_labels)  # Define CNNModel elsewhere
#         preprocess_batch = lambda b: b  # Inputs are tensors already
#     elif model_name == "mlp":
#         model = MLPModel(num_labels)  # Define MLPModel elsewhere
#         preprocess_batch = lambda b: b
#     elif model_name == "ast":
#         ast = ASTModelWrapper(num_labels)
#         model = ast.model
#         preprocess_batch = lambda b: b  # Already preprocessed via collate_fn
#     else:
#         raise ValueError("Unsupported model name")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = torch.nn.CrossEntropyLoss()

#     for epoch in range(epochs):
#         model.train()
#         for batch in train_loader:
#             inputs, labels = preprocess_batch(batch)
#             inputs = {k: v.to(device) for k, v in inputs.items()} if isinstance(inputs, dict) else inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(**inputs).logits if model_name == "ast" else model(inputs)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch+1} done. Loss: {loss.item():.4f}")

#     return model


# # from models.ast import ASTModelWrapper

# # def train_model(model_name, train_loader, val_loader, num_labels=12, epochs=10, lr=1e-4):
# #     if model_name == "cnn":
# #         model = CNNModel(num_labels=num_labels)  # Define elsewhere
# #     elif model_name == "mlp":
# #         model = MLPModel(num_labels=num_labels)  # Define elsewhere
# #     elif model_name == "ast":
# #         ast = ASTModelWrapper(num_labels=num_labels)
# #         model = ast.model
# #     else:
# #         raise ValueError("Model name must be 'cnn', 'mlp', or 'ast'.")

# #     import torch
# #     from torch import nn, optim

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model.to(device)

# #     optimizer = optim.Adam(model.parameters(), lr=lr)
# #     criterion = nn.CrossEntropyLoss()

# #     for epoch in range(epochs):
# #         model.train()
# #         for batch in train_loader:
# #             inputs, labels = batch
# #             inputs, labels = inputs.to(device), labels.to(device)

# #             outputs = model(**inputs).logits if model_name == "ast" else model(inputs)
# #             loss = criterion(outputs, labels)

# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()

# #         print(f"Epoch {epoch + 1}/{epochs} completed.")

# #     return model

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, model_name, train_loader, val_loader=None, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if model_name == "ast":
                outputs = model(**inputs).logits
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    return model
