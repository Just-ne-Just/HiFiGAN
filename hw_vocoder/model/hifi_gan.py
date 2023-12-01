from torch import nn
import torch
from hw_vocoder.model.descriminator import Descriminator
from hw_vocoder.model.generator import Generator

class HiFiGAN(nn.Module):
    def __init__(self, u_kernel_size, res_kernel_size, dilation, stride, conv_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.generator = Generator(u_kernel_size, res_kernel_size, dilation, stride, conv_channels)
        self.descriminator = Descriminator()
    
    def generate(self, x, **kwargs):
        return self.generator(x)
    
    def descriminate(self, gen, real, **kwargs):
        return self.descriminator(gen, real)
    
    def forward(self, x):
        return x
    

        
