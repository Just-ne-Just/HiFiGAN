import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

def get_padding(k, s):
    return (k * s - s) // 2

class PeriodDiscriminator(nn.Module):
    def __init__(self, kernel_size, stride, period, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
                nn.LeakyReLU()
            ),
        ])
        
        self.last_conv = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.period = period
    
    def forward(self, x):
        features = []
        if x.shape[-1] % self.period:
            x = nn.functional.pad(x, (0, self.period - x.shape[-1] % self.period), "reflect")
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // self.period, self.period)
        for conv in self.conv_blocks:
            x = conv(x)
            features.append(x)
        
        x = self.last_conv(x)
        features.append(x)
        return x.reshape(x.shape[0], -1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.discriminators = nn.ModuleList()

        for period in [2, 3, 5, 7, 11]:
            self.discriminators.add_module(f"disc_{period}", PeriodDiscriminator(5, 3, period))
    
    def forward(self, x):
        x_periods = []
        features = []
        for discriminator in self.discriminators:
            x_period, feature = discriminator(x)
            x_periods.append(x_period)
            features.append(feature)
        return x_periods, features


class ScaleDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
                nn.LeakyReLU()
            )
        ])
        self.last_conv = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
    
    def forward(self, x):
        features = []
        for conv in self.conv_blocks:
            x = conv(x)
            features.append(x)
        
        x = self.last_conv(x)
        features.append(x)
        return x.reshape(x.shape[0], -1), features
    

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            nn.Sequential(
                nn.AvgPool1d(4, 2, padding=2),
                ScaleDiscriminator(),
            ),
            nn.Sequential(
                nn.AvgPool1d(4, 2, padding=2),
                ScaleDiscriminator(),
            ),
        ])

    def forward(self, x):
        x_scales = []
        features = []
        for discriminator in self.discriminators:
            x_scale, feature = discriminator(x)
            x_scales.append(x_scale)
            features.append(feature)
        return x_scales, features


class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, gen, real):
        if gen.shape[-1] > real.shape[-1]:
            real = nn.ConstantPad3d(padding=(0, 0, 0, 0, 0, gen.shape[-1] - real.shape[-1]), value=0)(real)
            # real = nn.functional.pad(real, pad=(0, 0, gen.shape[-1] - real.shape[-1]), value=0)

        period_real, period_feature_real = self.mpd(real)
        period_generated, period_feature_generated = self.mpd(gen)
        scale_real, scale_feature_real = self.msd(real)
        scale_generated, scale_feature_generated = self.msd(gen)
        
        return {
            "period_real": period_real,
            "period_feature_real": period_feature_real,
            "period_generated": period_generated,
            "period_feature_generated": period_feature_generated,
            "scale_real": scale_real,
            "scale_feature_real": scale_feature_real,
            "scale_generated": scale_generated,
            "scale_feature_generated": scale_feature_generated
        }