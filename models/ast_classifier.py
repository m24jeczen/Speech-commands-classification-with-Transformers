import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTForAudioClassification

class ASTClassifier(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=35):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(model_name, num_labels=num_labels)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs.logits

    def extract_features(self, waveform, sampling_rate=16000):
        # Expect waveform as NumPy array or Tensor: [samples]
        inputs = self.feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        return inputs
