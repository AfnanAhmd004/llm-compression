"""Knowledge distillation: a smaller student learns from the teacher's softened distribution."""
from __future__ import annotations

import random

import torch
from torch.nn import functional as F

from .data import batches


def distill(teacher, student, text: str, steps: int = 1500, T: float = 2.0, alpha: float = 0.7, lr: float = 2e-3,
            batch_size: int = 32, seed: int = 0):
    """loss = alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(labels)."""
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    it = batches(text, student.cfg.block_size, batch_size, random.Random(seed))
    student.train()
    for _ in range(steps):
        x, y = next(it)
        with torch.no_grad():
            t_logits, _ = teacher(x)
        s_logits, ce = student(x, y)
        kd = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1), reduction="batchmean") * T * T
        loss = alpha * kd / x.shape[1] + (1 - alpha) * ce
        opt.zero_grad()
        loss.backward()
        opt.step()
    return student


def train_lm(model, text: str, steps: int, lr: float = 2e-3, batch_size: int = 32, seed: int = 0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    it = batches(text, model.cfg.block_size, batch_size, random.Random(seed))
    model.train()
    for _ in range(steps):
        x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
