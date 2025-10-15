# mlproject/engine/trainer.py
from __future__ import annotations
from typing import Dict, Any, Optional, Iterable
import torch
from torch.amp import GradScaler
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import numpy as np

from eps_seg.config.train import TrainConfig
from eps_seg.train.callbacks import Callback
from eps_seg.train.optimizers import _make_optimizer_and_scheduler

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Iterable,
        val_loader: Optional[Iterable],
        cfg: TrainConfig,
        *,
        callbacks: Optional[list[Callback]] = None,
        gaussian_noise_std: Optional[float] = None,
        directory_path: str = "./",
        monitor_key: str = "val_total",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.callbacks = callbacks or []
        self.gaussian_noise_std = gaussian_noise_std
        self.directory_path = directory_path
        self.monitor_key = monitor_key
        self.device = device or getattr(
            model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.optimizer, self.scheduler = _make_optimizer_and_scheduler(
            self.model, cfg.lr, 0.0
        )
        self.scaler = GradScaler(init_scale=cfg.gradient_scale, enabled=cfg.amp)

        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.bad_epochs = 0
        self.best_metric = np.inf
        self.extra_state: Dict[str, Any] = {
            "threshold": 0.50
        }

        cudnn.benchmark = True
        cudnn.fastest = True

    # ---- lifecycle dispatch
    def _dispatch(self, event: str, **kw):
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn:
                fn(self, **kw)

    def fit(self):
        self._dispatch("on_fit_start")
        for epoch in range(self.current_epoch, self.cfg.max_epochs):
            self.current_epoch = epoch
            self._dispatch("on_epoch_start", epoch=epoch)
            train_logs = self._train_one_epoch()
            if self.val_loader is not None:
                val_logs = self._validate()
                # plateau bookkeeping
                metric = float(val_logs.get(self.monitor_key, np.inf))
                if metric + 1e-6 < self.best_metric:
                    self.best_metric = metric
                    self.bad_epochs = 0
                else:
                    self.bad_epochs += 1
                self._dispatch("on_validation_end", logs=val_logs)
                if self.should_stop:
                    break
            self._dispatch("on_epoch_end", epoch=epoch)
            if self.should_stop:
                break
        self._dispatch("on_fit_end")

    # ---- one epoch train
    def _train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        log_interval = 5
        keys = [
            "IP",
            "KL",
            "CL",
            "CE",
            "EL",
            "Total",
            "tp",
            "tn",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ]
        running = {k: 0.0 for k in keys}

        for batch_idx, (x, y, z, _) in tqdm(
            enumerate(self.train_loader), desc="Training"
        ):
            # dataset may yield dims [1,B,...]
            x = x.squeeze(0).to(self.device, dtype=torch.float)
            y = y.squeeze(0).to(self.device, dtype=torch.float)
            if isinstance(z, torch.Tensor):
                z = z.squeeze(0)

            self.optimizer.zero_grad(set_to_none=True)

            threshold = float(self.extra_state.get("threshold", 0.50))
            print(f"TODO: threshold is not used in forward_pass")
            outputs = self.model.forward_pass(x, y, amp=self.cfg.amp)
            

            # UNSUP confusion-matrix metrics (only if provided like your code)
            if getattr(self.model, "training_mode", "") == "unsupervised" and isinstance(
                z, torch.Tensor
            ):
                pairs = [
                    (i, j) for i in range(x.shape[0]) for j in range(i + 1, x.shape[0])
                ]
                quadrants = outputs.get("q", {})
                center_y, center_x = 31, 31
                patch_labels = z[:, center_y, center_x]
                y_true, y_pred = [], []
                quadrant_expectation = {
                    "top_left": 0,
                    "top_right": 0,
                    "bottom_left": 1,
                    "bottom_right": 1,
                }
                for quadrant, pair_indices in quadrants.items():
                    for i, j in [pairs[k] for k in pair_indices.tolist()]:
                        y_true.append(quadrant_expectation[quadrant])
                        y_pred.append(
                            int(patch_labels[i].item() == patch_labels[j].item())
                        )
                import numpy as np
                from sklearn.metrics import (
                    precision_score,
                    recall_score,
                    f1_score,
                    confusion_matrix,
                )

                if len(y_true) > 0:
                    tn, fp, fn, tp = confusion_matrix(
                        y_true, y_pred, labels=[0, 1]
                    ).ravel()
                    running["tp"] += tp
                    running["tn"] += tn
                    running["fp"] += fp
                    running["fn"] += fn
                    running["precision"] += precision_score(
                        y_true, y_pred, zero_division=0
                    )
                    running["recall"] += recall_score(y_true, y_pred, zero_division=0)
                    running["f1"] += f1_score(y_true, y_pred, zero_division=0)

            inpainting_loss = outputs["inpainting_loss"]
            kl_loss = outputs["kl_loss"]
            cl_loss = outputs.get("cl_loss", torch.tensor(0.0, device=self.device))
            if torch.isnan(cl_loss):
                cl_loss = torch.tensor(0.0, device=self.device)
            ce = outputs.get("ce", torch.tensor(0.0, device=self.device))
            entropy = outputs.get("entropy", torch.tensor(0.0, device=self.device))

            loss = (
                self.cfg.alpha * inpainting_loss
                + self.cfg.beta * kl_loss
                + self.cfg.gamma * cl_loss
                + ce
                + entropy
            )

            self.scaler.scale(loss).backward()
            if self.cfg.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if hasattr(self.model, "increment_global_step"):
                self.model.increment_global_step()
            self.global_step += 1

            # accumulate
            running["IP"] += float(inpainting_loss.item() * self.cfg.alpha)
            running["KL"] += float(kl_loss.item() * self.cfg.beta)
            running["CL"] += float(cl_loss.item() * self.cfg.gamma)
            running["CE"] += float(ce.item())
            running["EL"] += float(entropy.item())
            running["Total"] += float(loss.item())

            if (batch_idx + 1) % log_interval == 0:
                avg = {k: v / log_interval for k, v in running.items()}
                self._dispatch("on_batch_end", batch_idx=batch_idx, logs=avg)
                running = {k: 0.0 for k in running}  # reset window

        return running

    # ---- validation
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        agg = {
            "val_IP": 0.0,
            "val_KL": 0.0,
            "val_CE": 0.0,
            "val_EL": 0.0,
            "val_CL": 0.0,
            "val_total": 0.0,
        }
        n_batches = 0
        for batch_idx, (x, y, z, _) in tqdm(
            enumerate(self.val_loader), desc="Validation"
        ):
            x = x.squeeze(0).to(self.device, dtype=torch.float)
            y = y.squeeze(0).to(self.device, dtype=torch.float)
            if isinstance(z, torch.Tensor):
                z = z.squeeze(0)
            outputs = self.model.validation_step(x, y, amp=self.cfg.amp)
            ip = outputs["inpainting_loss"]
            kl = outputs["kl_loss"]
            ce = outputs.get("ce", torch.tensor(0.0, device=self.device))
            en = outputs.get("entropy", torch.tensor(0.0, device=self.device))
            cl = outputs.get("cl_loss", torch.tensor(0.0, device=self.device))
            if torch.isnan(cl):
                cl = torch.tensor(0.0, device=self.device)
            total = (
                self.cfg.alpha * ip + self.cfg.beta * kl + self.cfg.gamma * cl + ce + en
            )

            agg["val_IP"] += float(self.cfg.alpha * ip)
            agg["val_KL"] += float(self.cfg.beta * kl)
            agg["val_CE"] += float(ce)
            agg["val_EL"] += float(en)
            agg["val_CL"] += float(self.cfg.gamma * cl)
            agg["val_total"] += float(total)
            n_batches += 1

        if n_batches > 0:
            for k in list(agg.keys()):
                agg[k] /= n_batches
        return agg
