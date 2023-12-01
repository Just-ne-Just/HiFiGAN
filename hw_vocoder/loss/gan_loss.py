import torch
from torch import Tensor
import torch.nn as nn
from hw_vocoder.spectrogram.spectrogram import MelSpectrogram

class GANLoss(nn.Module):
    def __init__(self, mel_scale, fm_scale):
        super().__init__()
        self.mel_scale = mel_scale
        self.fm_scale = fm_scale
        self.mel = MelSpectrogram()
          
    def forward(self, spectrogram, **kwargs) -> Tensor:
        