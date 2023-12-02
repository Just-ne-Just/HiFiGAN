import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
from hw_vocoder.spectrogram.spectrogram import MelSpectrogram
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    audio = []

    for item in dataset_items:
        audio.append(item['audio'][0])
        
    
    audio = pad_sequence(audio, batch_first=True)
    mel = MelSpectrogram()(audio)

    # print(audio.shape)
    # print(mel.shape)
        
    return {
        "audio": audio.unsqueeze(1),
        "spectrogram": mel
    }