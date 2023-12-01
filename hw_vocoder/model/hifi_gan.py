from torch import nn
import torch
from hw_vocoder.model.descriminator import Descriminator
from hw_vocoder.model.generator import Generator
import numpy as np

class HiFiGAN(nn.Module):
    def __init__(self, u_kernel_size, res_kernel_size, dilation, stride, conv_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.generator = Generator(u_kernel_size, res_kernel_size, dilation, stride, conv_channels)
        self.descriminator = Descriminator()
    
    def generate(self, spectrogram, **kwargs):
        return self.generator(spectrogram)
    
    def descriminate(self, gen, real, **kwargs):
        return self.descriminator(gen, real)
    
    def forward(self, x):
        return x
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    

        
