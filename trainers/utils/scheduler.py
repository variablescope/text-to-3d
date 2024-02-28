import torch.nn as nn
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch


def get_scheduler(cfg, optimizer):
    if cfg.lr_scheduler.name == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg.lr_scheduler.params.warmup_steps)
    elif cfg.lr_scheduler.name == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg.lr_scheduler.params.warmup_steps, num_training_steps=cfg.training.num_epochs*len(train_dl))
