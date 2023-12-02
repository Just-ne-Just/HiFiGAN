import torch
from torch import Tensor
import torch.nn as nn
from hw_vocoder.spectrogram.spectrogram import MelSpectrogram

class DAdvLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, target_out, gen_out, **kwargs):
        gen_loss = 0
        target_loss = 0

        for out in target_out:
            target_loss = target_loss + torch.mean((out - 1) ** 2, dim=-1)

        for out in gen_out:
            gen_loss = gen_loss + torch.mean(out ** 2, dim=-1)

        return target_loss + gen_loss


class GAdvLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_out, **kwargs):
        gen_loss = 0
        for out in gen_out:
            gen_loss = gen_loss + torch.mean((out - 1) ** 2, dim=-1)
        return gen_loss.sum()


class MelLoss(nn.Module):
    def __init__(self, mel_scale, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mel_scale = mel_scale
    
    def forward(self, mel_gen, mel_target, **kwargs):
        mel_loss = torch.nn.functional.l1_loss(mel_target, mel_gen)
        return self.mel_scale * mel_loss


class FMLoss(nn.Module):
    def __init__(self, fm_scale, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fm_scale = fm_scale
    
    def forward(self, target_features, gen_features, **kwargs):
        fm_loss = 0
        for pred, gen in zip(target_features, gen_features):
            fm_loss = fm_loss + torch.nn.functional.l1_loss(gen, pred)
        
        return self.fm_scale * fm_loss


class GANLoss(nn.Module):
    def __init__(self, mel_scale, fm_scale):
        super().__init__()
        self.dadv_loss = DAdvLoss()
        self.gadv_loss = GAdvLoss()
        self.mel_loss = MelLoss(mel_scale)
        self.fm_loss = FMLoss(fm_scale)
        self.mel = MelSpectrogram()
          
    def forward(self, 
                spectrogram,
                gen_audio,
                period_generated,
                period_feature_generated,
                period_real,
                period_feature_real,
                scale_generated,
                scale_feature_generated,
                scale_real,
                scale_feature_real,
                **kwargs) -> Tensor:
        
        gen_spectrogram = self.mel(gen_audio.squeeze(1))
        spectrogram = nn.ConstantPad3d(padding=(0, 0, 0, 0, 0, gen_spectrogram.shape[-1] - spectrogram.shape[-1]), value=0)(spectrogram)

        gadv_loss = self.gadv_loss(period_generated)
        gadv_loss = gadv_loss + self.gadv_loss(scale_generated)

        dadv_loss = self.dadv_loss(period_real, period_generated)
        dadv_loss = dadv_loss + self.dadv_loss(scale_real, scale_generated)

        fm_loss = self.fm_loss(period_feature_real, period_feature_generated)
        fm_loss = fm_loss + self.fm_loss(scale_feature_real, scale_feature_generated)

        mel_loss = self.mel_loss(gen_spectrogram, spectrogram)

        generator_loss = gadv_loss + fm_loss + mel_loss
        descriminator_loss = dadv_loss

        return generator_loss, descriminator_loss, gadv_loss, fm_loss, mel_loss

