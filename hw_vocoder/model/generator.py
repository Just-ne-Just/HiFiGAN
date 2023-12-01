import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ResBlock(nn.Module):
    def __init__(self, kernel_size, dilation, conv_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.blocks = nn.ModuleList()
        
        for m in range(len(dilation)):
            tmp_list = nn.ModuleList()
            for l in range(len(dilation[m])):
                small_block = nn.Sequential(
                    nn.LeakyReLU(),
                    weight_norm(nn.Conv2d(conv_channels, 
                                          conv_channels, 
                                          (kernel_size, 1), 
                                          dilation=dilation[m][l],
                                          padding=(kernel_size * dilation[m][l] - dilation[m][l]) // 2))
                )
                tmp_list.add_module(f"block_{(m + 1) * (l + 1)}", small_block)
            self.blocks.add_module(f"block_list_{m + 1}", tmp_list)
        
        def forward(self, x):
            for list_block in self.blocks:
                x_old = x.clone()
                for block in list_block:
                    x = block(x) 
                x = x + x_old
            return x

class MRF(nn.Module):
    def __init__(self, kernel_size, dilation, conv_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert len(kernel_size) == len(dilation)
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            self.blocks.add_module(f"resblock_{i + 1}", ResBlock(kernel_size[i], dilation[i], conv_channels))
        
    def forward(self, x):
        s = None
        for block in self.blocks:
            out = block(x)
            s = (s + out) if s is not None else out
        return s
        
class Generator(nn.Module):
    def __init__(self, u_kernel_size, res_kernel_size, dilation, stride, conv_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.first_conv = weight_norm(nn.Conv1d(80, conv_channels, 7, 1, padding=3))
        self.mrf_blocks = nn.ModuleList()
        self.len = len(u_kernel_size)
        for l in range(len(u_kernel_size)):
            mrf_conv_channels = conv_channels // (2**(l + 1))
            block = nn.Sequential(
                nn.LeakyReLU(),
                nn.ConvTranspose1d(conv_channels // (2**l), 
                                   conv_channels // (2**(l + 1)), 
                                   u_kernel_size[l], 
                                   stride[l], 
                                   padding=(u_kernel_size[l] - stride[l]) // 2),
                MRF(res_kernel_size, dilation, mrf_conv_channels)
            )
            self.mrf_blocks.add_module(f"mrf_block_{l + 1}", block)
        
        self.out = nn.Sequential(
            nn.LeakyReLU(),
            weight_norm(nn.Conv1d(mrf_conv_channels, 1, 7, 1, padding=3)),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.first_conv(x)
        for mrf_block in self.mrf_blocks:
            x = mrf_block(x)
        x = self.out(x)
        return x


