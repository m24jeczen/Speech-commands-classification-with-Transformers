import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

class ASTCollateFn:
    def __init__(self, feature_extractor, min_length=400, sample_rate=16000):
        self.feature_extractor = feature_extractor
        self.min_length = min_length
        self.sample_rate = sample_rate

    def __call__(self, batch):
        input_values = []
        labels = []

        for waveform, label in batch:
            if waveform.shape[0] < self.min_length:
                waveform = pad(waveform, (0, self.min_length - waveform.shape[0]))

            waveform = waveform.numpy()
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )["input_values"][0]
            input_values.append(inputs)
            labels.append(label)

        padded_inputs = pad_sequence(input_values, batch_first=True)
        return {"input_values": padded_inputs}, torch.tensor(labels)
