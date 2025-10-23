import os
import sys
import json
import math
import argparse
import logging
import subprocess
import glob

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import your own modules
import commons
import utils
from data_utils import TextAudioSpeakerEmotionLoader, TextAudioSpeakerEmotionCollate, DistributedBucketSampler
from emotive_models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

# Enable CuDNN benchmark for speed.
torch.backends.cudnn.benchmark = True

global_step = 0  # Global training step

# --- Monkey-patch resource_tracker (if on Linux) ---
if sys.platform.startswith("linux"):
    try:
        import multiprocessing.resource_tracker as resource_tracker
        def _ignore(*args, **kwargs):
            pass
        resource_tracker.register = _ignore
        resource_tracker.unregister = _ignore
    except Exception as e:
        print("Resource tracker patch failed:", e)

# --- Main Training Script Functions ---

def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '7173'  # Choose an available port

    hps = utils.get_hparams()  # Loads and merges the JSON config with command-line overrides.
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    # Only rank 0 creates the logger and summary writers.
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info("Hyperparameters:\n%s", hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    else:
        logger = None
        writer = None
        writer_eval = None

    # Initialize the distributed process group.
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # --- Data Loading ---
    train_dataset = TextAudioSpeakerEmotionLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        boundaries=[32, 100, 200, 300, 400, 500, 600, 700, 800],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )
    collate_fn = TextAudioSpeakerEmotionCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,  # Use 0 workers for stability; later you can try increasing this.
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler
    )

    if rank == 0:
        eval_dataset = TextAudioSpeakerEmotionLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )
    else:
        eval_loader = None

    # --- Model Initialization ---
    # Extract model parameters from hps.model; remove keys not expected by SynthesizerTrn.
    model_params = hps.model.__dict__.copy()
    # Remove extraneous keys (adjust this list as needed)
    for key in ['data', 'model', 'n_layers_q', 'use_spectral_norm']:
        model_params.pop(key, None)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_emotions=hps.data.num_emotions,
        **model_params
    ).cuda(rank)

    # For discriminator, if you need use_spectral_norm you can pass it separately from hps.model
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    # --- Load Pretrained Checkpoint (if any) ---
    if hps.train.pretrained_model:
        # Use the underlying module if already wrapped
        model_to_load = net_g.module if hasattr(net_g, 'module') else net_g
        utils.load_checkpoint(hps.train.pretrained_model, model_to_load, None, strict=False)

    # --- Optimizers and Schedulers ---
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    # Wrap models in DDP.
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    # Optionally, load latest checkpoints if available.
    try:
        ckpt_g = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        ckpt_d = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        if ckpt_g:
            _, _, _, epoch_str = utils.load_checkpoint(ckpt_g, net_g, optim_g, strict=False)
        if ckpt_d:
            _, _, _, epoch_str = utils.load_checkpoint(ckpt_d, net_d, optim_d, strict=False)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        if rank == 0 and logger:
            logger.info("No checkpoint loaded; starting from scratch. Exception: %s", str(e))
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    scaler_g = GradScaler(enabled=hps.train.fp16_run)
    scaler_d = GradScaler(enabled=hps.train.fp16_run)

    # --- Training Loop ---
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(rank, epoch, hps, [net_g, net_d],
                           [optim_g, optim_d],
                           [scheduler_g, scheduler_d],
                           [scaler_g, scaler_d],
                           [train_loader, eval_loader],
                           logger, [writer, writer_eval])
        if epoch % hps.train.lr_decay_interval == 0:
            scheduler_g.step()
            scheduler_d.step()

    # Clean-up
    if rank == 0 and writer:
        writer.close()
    if rank == 0 and writer_eval:
        writer_eval.close()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scalers, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scaler_g, scaler_d = scalers
    train_loader, eval_loader = loaders

    if writers is not None:
        writer, writer_eval = writers

    # Set the sampler epoch for reproducibility.
    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emotions) in enumerate(train_loader):
        x = x.cuda(rank, non_blocking=True)
        x_lengths = x_lengths.cuda(rank, non_blocking=True)
        spec = spec.cuda(rank, non_blocking=True)
        spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
        y = y.cuda(rank, non_blocking=True)
        y_lengths = y_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)
        emotions = emotions.cuda(rank, non_blocking=True)

        # Forward pass through generator.
        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                x, x_lengths, spec, spec_lengths,
                sid=speakers,
                emotion_labels=emotions
            )

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

        # Discriminator loss.
        with autocast(enabled=False):
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler_d.scale(loss_disc_all).backward()
        scaler_d.unscale_(optim_d)
        grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), hps.train.grad_clip)
        scaler_d.step(optim_d)
        scaler_d.update()

        # Generator loss.
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        optim_g.zero_grad()
        scaler_g.scale(loss_gen_all).backward()
        scaler_g.unscale_(optim_g)
        grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), hps.train.grad_clip)
        scaler_g.step(optim_g)
        scaler_g.update()

        # Logging and summaries on rank 0.
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc.item(), loss_gen.item(), loss_fm.item(), loss_mel.item(), loss_dur.item(), loss_kl.item()]
                logger.info(f"Epoch: {epoch} [{batch_idx}/{len(train_loader)}] Losses: {losses} LR: {lr:.6f}")

                scalar_dict = {
                    "loss/g/total": loss_gen_all.item(),
                    "loss/d/total": loss_disc_all.item(),
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss/g/fm": loss_fm.item(),
                    "loss/g/mel": loss_mel.item(),
                    "loss/g/dur": loss_dur.item(),
                    "loss/g/kl": loss_kl.item()
                }
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].T.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].T.cpu().numpy()),
                    "all/attn": utils.plot_alignment_to_numpy(attn[0,0].cpu().numpy())
                }
                utils.summarize(writer=writer, global_step=global_step, scalars=scalar_dict, images=image_dict)
            if global_step % hps.train.eval_interval == 0 and eval_loader is not None:
                evaluate(hps, net_g, eval_loader, writer_eval)
                # Save checkpoints
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                        os.path.join(hps.model_dir, f"G_{global_step}.pth"))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                        os.path.join(hps.model_dir, f"D_{global_step}.pth"))
        global_step += 1

    if rank == 0:
        logger.info(f"Epoch {epoch} completed")


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emotions) in enumerate(eval_loader):
            x = x.cuda(0)
            x_lengths = x_lengths.cuda(0)
            spec = spec.cuda(0)
            spec_lengths = spec_lengths.cuda(0)
            y = y.cuda(0)
            y_lengths = y_lengths.cuda(0)
            speakers = speakers.cuda(0)
            emotions = emotions.cuda(0)
            # Use only first sample for evaluation.
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            emotions = emotions[:1]
            y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, sid=speakers, emotion_labels=emotions, max_len=1000)
            y_hat_lengths = (mask.sum([1,2]).long() * hps.data.hop_length)
            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, 
                                      hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            break

    image_dict = {"gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].T.cpu().numpy())}
    audio_dict = {"gen/audio": y_hat[0, :, :y_hat_lengths[0]]}
    if global_step == 0:
        image_dict["gt/mel"] = utils.plot_spectrogram_to_numpy(mel[0].T.cpu().numpy())
        audio_dict["gt/audio"] = y[0, :, :y_lengths[0]]
    utils.summarize(writer=writer_eval, global_step=global_step, images=image_dict, audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)
    generator.train()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
