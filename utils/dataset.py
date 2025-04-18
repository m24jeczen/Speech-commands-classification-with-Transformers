# import os
# from glob import glob
# import torch
# from torch.utils.data import Dataset
# from torchaudio import load
# from sklearn.preprocessing import LabelEncoder
# from models.ast import ASTModelWrapper
# from config import SAMPLE_RATE

# import os
# from glob import glob
# import torch
# from torch.utils.data import Dataset
# from torchaudio import load
# from sklearn.preprocessing import LabelEncoder
# from transformers import ASTFeatureExtractor
# from config import SAMPLE_RATE
# from utils.audio_utils import create_mel_spectrogram  # if using for CNN

# class SpeechDataset(Dataset):
#     def __init__(self, audio_dir, labels, mode="ast"):
#         self.paths = []
#         self.labels = []
#         for label in labels:
#             files = glob(os.path.join(audio_dir, label, "*.wav"))
#             self.paths.extend(files)
#             self.labels.extend([label] * len(files))

#         self.encoder = LabelEncoder()
#         self.encoded_labels = self.encoder.fit_transform(self.labels)
#         self.mode = mode

#         if self.mode == "ast":
#             self.feature_extractor = ASTFeatureExtractor.from_pretrained(
#                 "MIT/ast-finetuned-audioset-10-10-0.4593"
#             )

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         waveform, sr = load(self.paths[idx])
#         waveform = waveform.squeeze(0)
#         label = self.encoded_labels[idx]

#         if self.mode == "ast":
#             return waveform, label

#         elif self.mode == "cnn":
#             mel_spec = create_mel_spectrogram(self.paths[idx])
#             return mel_spec, label

#         elif self.mode == "mlp":
#             embedding = waveform.mean(dim=-1, keepdim=True)
#             return embedding, label


#         else:
#             raise ValueError("Unsupported mode. Choose from 'ast', 'cnn', 'mlp'.")


# import torch
# from torch.nn.utils.rnn import pad_sequence
# from models.ast import ASTModelWrapper
# from config import SAMPLE_RATE

# def ast_collate_fn(batch):
#     feature_extractor = ASTModelWrapper().feature_extractor
#     input_values = []
#     labels = []

#     for waveform, label in batch:
#         # Ensure waveform is long enough
#         if waveform.shape[0] < 400:
#             pad_len = 400 - waveform.shape[0]
#             waveform = torch.nn.functional.pad(waveform, (0, pad_len))

#         waveform = waveform.numpy()
#         inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"][0]
#         input_values.append(inputs)
#         labels.append(label)

#     padded_inputs = pad_sequence(input_values, batch_first=True)
#     return {"input_values": padded_inputs}, torch.tensor(labels)

import os
from glob import glob
import torch
from torch.utils.data import Dataset
from torchaudio import load
from sklearn.preprocessing import LabelEncoder
from config import SAMPLE_RATE
from utils.audio_utils import create_mel_spectrogram


class SpeechDataset(Dataset):
    def __init__(self, audio_dir, labels, mode="ast"):
        self.paths = []
        self.labels = []
        for label in labels:
            files = glob(os.path.join(audio_dir, label, "*.wav"))
            self.paths.extend(files)
            self.labels.extend([label] * len(files))

        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform, sr = load(self.paths[idx])
        waveform = waveform.squeeze(0)
        label = self.encoded_labels[idx]

        if self.mode == "ast":
            return waveform, label  # no AST processing here!

        elif self.mode == "cnn":
            mel_spec = create_mel_spectrogram(self.paths[idx])
            return mel_spec, label

        elif self.mode == "mlp":
            embedding = waveform.mean(dim=-1, keepdim=True)
            return embedding, label

        else:
            raise ValueError("Unsupported mode. Use 'cnn', 'mlp', or 'ast'.")
