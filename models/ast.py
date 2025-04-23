# models/ast.py
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch.nn as nn

class ASTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ASTClassifier, self).__init__()
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # 👈 Fix the mismatch error
        )

    def forward(self, input_values):
        return self.model(input_values).logits
