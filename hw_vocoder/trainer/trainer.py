import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_vocoder.base import BaseTrainer
from hw_vocoder.base.base_text_encoder import BaseTextEncoder
from hw_vocoder.logger.utils import plot_spectrogram_to_buf
from hw_vocoder.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            gen_optimizer,
            desc_optimizer,
            config,
            device,
            dataloaders,
            gen_lr_scheduler=None,
            desc_lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, gen_optimizer, desc_optimizer, config, device, gen_lr_scheduler, desc_lr_scheduler)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "generator_loss", 
            "descriminator_loss", 
            "gadv_loss", 
            "fm_loss", 
            "mel_loss", 
            "grad_norm", 
            writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        bar = tqdm(range(self.len_epoch), desc='train')

        for batch_idx, batch in enumerate(
                self.train_dataloader
        ):
            bar.update(1)
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                print("ERROR")
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    raise e
                    # continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} GLoss: {:.6f} DLoss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["generator_loss"].item(), batch["descriminator_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate gen", self.gen_lr_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning rate desc", self.desc_lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            
            # print(batch_idx, self.len_epoch)
            if batch_idx + 1 >= self.len_epoch:
                # print("BREAK")
                break
        log = last_train_metrics

        self._log_audio(batch['gen_audio'][0], 22050, 'train.wav')

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.gen_optimizer.zero_grad()
            self.desc_optimizer.zero_grad()
        gen_outputs = self.model.generate(**batch)
        batch.update(gen_outputs)

        desc_outputs = self.model.descriminate(gen=batch["gen_audio"].detach(), real=batch["audio"])
        batch.update(desc_outputs)

        if is_train:
            generator_loss, descriminator_loss, gadv_loss, fm_loss, mel_loss = self.criterion(**batch)
            descriminator_loss.backward()
            self._clip_grad_norm()
            self.desc_optimizer.step()

            desc_outputs = self.model.descriminate(gen=batch["gen_audio"].detach(), real=batch["audio"])
            batch.update(desc_outputs)
            generator_loss, descriminator_loss, gadv_loss, fm_loss, mel_loss = self.criterion(**batch)
            generator_loss.backward()
            self._clip_grad_norm()
            self.gen_optimizer.step()

            batch["generator_loss"] = generator_loss
            batch["descriminator_loss"] = descriminator_loss
            batch["gadv_loss"] = gadv_loss
            batch["fm_loss"] = fm_loss 
            batch["mel_loss"] = mel_loss

            metrics.update("generator_loss", generator_loss.item())
            metrics.update("descriminator_loss", descriminator_loss.item())
            metrics.update("gadv_loss", gadv_loss.item())
            metrics.update("fm_loss", fm_loss.item())
            metrics.update("mel_loss", mel_loss.item())
            metrics.update("grad_norm", self.get_grad_norm())

            if self.gen_lr_scheduler is not None:
                self.gen_lr_scheduler.step()
            
            if self.desc_lr_scheduler is not None:
                self.desc_lr_scheduler.step()
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch, train=False)
            self._log_spectrogram(batch["spectrogram"])
            self._log_audio(batch["audio"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            log_probs,
            log_probs_length,
            audio_path,
            examples_to_log=10,
            train=False,
            *args,
            **kwargs,
    ):
        # TODO: implement logging of beam search results
        if self.writer is None:
            return
        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        all_hypos = [""] * len(argmax_texts)
        if not train:
            all_hypos = self.text_encoder.ctc_lm(log_probs, log_probs_length, 3)

        tuples = list(zip(all_hypos, argmax_texts, text, argmax_texts_raw, audio_path))
        shuffle(tuples)
        rows = {}
        for lm_pred, pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = BaseTextEncoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            lm_wer = calc_wer(target, lm_pred) * 100
            lm_cer = calc_cer(target, lm_pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
                "lm prediction": lm_pred,
                "lm wer": lm_wer,
                "lm cer": lm_cer,
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))
    
    def _log_audio(self, audio, sample_rate, name):
        # spectrogram = random.choice(spectrogram_batch.cpu())
        # image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_audio(name, audio, sample_rate=sample_rate)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
