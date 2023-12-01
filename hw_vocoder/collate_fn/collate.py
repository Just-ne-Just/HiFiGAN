import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    audio = []
    spec = []
    spec_length = []
    text_encoded = []
    text_encoded_length = []
    text = []
    audio_path = []

    for item in dataset_items:
        audio.append(item['audio'][0])
        spec.append(item['spectrogram'][0].T)
        spec_length.append(item['spectrogram'].shape[2])
        text_encoded.append(item['text_encoded'][0])
        text_encoded_length.append(len(item['text_encoded'][0]))
        text.append(item['text'])
        audio_path.append(item['audio_path'])
    
    audio = pad_sequence(audio, batch_first=True)
    spec = pad_sequence(spec, batch_first=True).transpose(1, 2)
    text_encoded = pad_sequence(text_encoded, batch_first=True)
        
    return {
        "audio": audio,
        "spectrogram": spec,
        "spectrogram_length": torch.tensor(spec_length),
        "text_encoded": text_encoded,
        "text_encoded_length": torch.tensor(text_encoded_length),
        "text": text,
        "audio_path": audio_path
    }
