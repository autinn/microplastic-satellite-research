# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import torch


class WarmupReduceLROnPlateauScheduler:
    """
    Minimal warmup + ReduceLROnPlateau wrapper with no external deps.

    - Linearly increases LR from init_lr to peak_lr over warmup_steps calls to step().
    - After warmup, delegates to torch.optim.lr_scheduler.ReduceLROnPlateau, called
      only when is_end_epoch=True with the given validation loss.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        peak_lr: float,
        warmup_steps: int,
        patience: int = 1,
        factor: float = 0.3,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.warmup_steps = int(max(0, warmup_steps))
        self.update_steps = 0
        self.factor = factor
        self.min_lr = min_lr

        # Set initial LR
        for group in self.optimizer.param_groups:
            group["lr"] = self.init_lr

        self.after_warmup = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience, factor=factor, min_lr=min_lr
        )

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, val_loss: Optional[float] = None, is_end_epoch: bool = False):
        # Warmup region
        if self.update_steps < self.warmup_steps:
            progress = (self.update_steps + 1) / max(1, self.warmup_steps)
            lr_now = self.init_lr + progress * (self.peak_lr - self.init_lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr_now
        else:
            # Plateau scheduler called only at epoch end
            if is_end_epoch and val_loss is not None:
                self.after_warmup.step(val_loss)

        self.update_steps += 1
        return self.get_lr()
