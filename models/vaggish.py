import torch
import torch.nn as nn

from torchvggish import vggish, vggish_input


class VGGishClassifier(nn.Module):
    def __init__(self, num_classes=10, freeze_vggish=True):
        super().__init__()
        self.vgg = vggish()
        if freeze_vggish:
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        embeddings = []
        for waveform in x:
            waveform_np = waveform.squeeze().cpu().numpy()
            mel_input = vggish_input.waveform_to_examples(waveform_np, 16000)
            mel_input_tensor = torch.tensor(mel_input).float().to(x.device)

            with torch.no_grad():
                embed = self.vgg(mel_input_tensor)  # (N, 128)
                if embed.dim() == 1:
                    embed = embed.unsqueeze(0)
                elif embed.dim() > 2:
                    embed = embed.mean(dim=tuple(range(1, embed.dim())))

            embeddings.append(embed.mean(dim=0))

        features = torch.stack(embeddings)  # (batch_size, 128)

        if features.dim() == 1:
            features = features.unsqueeze(0)

        out = self.classifier(features)
        return out

