"""Official-like frozen REVE token probe."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TorchProbeResult:
    model: TorchTokenPredictor
    selected_epoch: int
    validation_balanced_accuracy: float
    validation_consistency_loss: float
    validation_prediction_disagreement: float
    history: tuple[dict[str, float], ...]
    trainable_parameters: int = 0
    auxiliary_parameters: int = 0


class TorchTokenPredictor:
    """Small sklearn-like prediction adapter around a PyTorch probe."""

    def __init__(self, module, *, device: str, batch_size: int) -> None:
        self.module = module
        self.device = device
        self.batch_size = batch_size

    def _logits(self, x: NDArray[np.floating]) -> NDArray[np.float32]:
        import torch

        self.module.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], self.batch_size):
                batch = torch.from_numpy(np.asarray(x[start : start + self.batch_size])).to(
                    self.device, dtype=torch.float32
                )
                chunks.append(self.module(batch).float().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def predict_logits(self, x: NDArray[np.floating]) -> NDArray[np.float32]:
        """Return raw class logits for post-hoc calibration audits."""

        return self._logits(x)

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.int64]:
        return self._logits(x).argmax(axis=1).astype(np.int64)

    def predict_proba(self, x: NDArray[np.floating]) -> NDArray[np.float32]:
        logits = self._logits(x)
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return probability / probability.sum(axis=1, keepdims=True)


def configure_torch_determinism(seed: int, *, strict: bool) -> None:
    """Configure deterministic probe training before CUDA is initialized."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = strict
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(strict, warn_only=False)


def fit_reve_token_probe(
    train_x: NDArray[np.floating],
    train_y: NDArray[np.integer],
    val_x: NDArray[np.floating],
    val_y: NDArray[np.integer],
    *,
    initial_query: NDArray[np.floating],
    n_classes: int,
    seed: int,
    device: str,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    dropout: float = 0.1,
    warmup_epochs: int = 5,
    patience: int = 5,
    clip_grad: float = 2.0,
    deterministic: bool = False,
    objective: str = "car_only",
    consistency_weight: float = 0.0,
) -> TorchProbeResult:
    """Fit a frozen-REVE probe with optional paired multi-view consistency.

    Inputs may be ``(trials, tokens, dim)`` for the original CAR-only probe or
    ``(trials, views, tokens, dim)`` for multi-reference objectives. View zero
    must be CAR so clean validation remains explicitly observable.
    """
    try:
        import torch
        from sklearn.metrics import balanced_accuracy_score
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        message = 'Token probe dependencies are missing. Install with: pip install -e ".[reve]"'
        raise RuntimeError(message) from exc

    objective = objective.casefold()
    allowed_objectives = {"car_only", "multi_view_ce", "rule_consistency"}
    if objective not in allowed_objectives:
        message = f"Unknown probe objective {objective!r}; expected one of {sorted(allowed_objectives)}"
        raise ValueError(message)
    if train_x.ndim not in {3, 4} or val_x.ndim != train_x.ndim:
        raise ValueError("Token features must be 3D CAR-only arrays or aligned 4D multi-view arrays")
    if train_x.ndim == 3:
        train_x = train_x[:, None, :, :]
        val_x = val_x[:, None, :, :]
    if train_x.shape[1] < 2 and objective != "car_only":
        raise ValueError(f"{objective} requires at least two aligned reference views")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight must be non-negative")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the REVE token probe but is unavailable")

    configure_torch_determinism(seed, strict=deterministic)

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, value):
            normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return normalized.type_as(value) * self.weight

    class ReveTokenHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            n_tokens, embed_dim = train_x.shape[-2:]
            query = torch.as_tensor(initial_query, dtype=torch.float32).reshape(1, 1, embed_dim)
            self.query = nn.Parameter(query.clone())
            flat_dim = (n_tokens + 1) * embed_dim
            self.norm = RMSNorm(flat_dim)
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(flat_dim, n_classes)
            self.scale = embed_dim**-0.5

        def forward(self, tokens):
            query = self.query.expand(tokens.shape[0], -1, -1)
            weights = torch.softmax(torch.matmul(query, tokens.transpose(-1, -2)) * self.scale, dim=-1)
            context = torch.matmul(weights, tokens)
            flattened = torch.cat([context, tokens], dim=1).flatten(1)
            return self.linear(self.dropout(self.norm(flattened)))

    model = ReveTokenHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(train_x)), torch.from_numpy(np.asarray(train_y))),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=str(device).startswith("cuda"),
    )
    predictor = TorchTokenPredictor(model, device=device, batch_size=batch_size)
    total_warmup_steps = warmup_epochs * len(train_loader)
    global_step = 0
    best_score = -1.0
    best_epoch = 0
    best_consistency = float("nan")
    best_disagreement = float("nan")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_ce = 0.0
        epoch_consistency = 0.0
        for tokens, labels in train_loader:
            tokens = tokens.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if global_step < total_warmup_steps:
                ratio = (10 ** (global_step / total_warmup_steps) - 1) / 9
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * max(ratio, 1e-3)
            optimizer.zero_grad(set_to_none=True)
            batch_size_actual, n_views = tokens.shape[:2]
            logits = model(tokens.flatten(0, 1)).reshape(batch_size_actual, n_views, n_classes)
            if objective == "car_only":
                ce_loss = criterion(logits[:, 0], labels)
            else:
                repeated_labels = labels[:, None].expand(-1, n_views).reshape(-1)
                ce_loss = criterion(logits.reshape(-1, n_classes), repeated_labels)

            consistency_loss = logits.new_zeros(())
            if objective == "rule_consistency":
                log_probabilities = torch.log_softmax(logits, dim=-1)
                probabilities = log_probabilities.exp()
                mean_probability = probabilities.mean(dim=1, keepdim=True).clamp_min(1e-8)
                consistency_loss = (
                    (probabilities * (log_probabilities - mean_probability.log())).sum(dim=-1).mean()
                )
            loss = ce_loss + consistency_weight * consistency_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
            total_loss += float(loss.item()) * labels.shape[0]
            epoch_ce += float(ce_loss.item()) * labels.shape[0]
            epoch_consistency += float(consistency_loss.item()) * labels.shape[0]
            global_step += 1

        validation_probabilities = np.stack(
            [predictor.predict_proba(val_x[:, view_index]) for view_index in range(val_x.shape[1])],
            axis=1,
        )
        validation_predictions = validation_probabilities.argmax(axis=-1)
        validation_scores = [
            float(balanced_accuracy_score(val_y, validation_predictions[:, view_index]))
            for view_index in range(val_x.shape[1])
        ]
        clipped_probability = np.clip(validation_probabilities, 1e-8, 1.0)
        mean_probability = np.clip(clipped_probability.mean(axis=1, keepdims=True), 1e-8, 1.0)
        validation_consistency = float(
            np.mean(
                np.sum(
                    clipped_probability * (np.log(clipped_probability) - np.log(mean_probability)),
                    axis=-1,
                )
            )
        )
        validation_disagreement = float(
            np.mean(np.any(validation_predictions != validation_predictions[:, :1], axis=1))
        )
        clean_score = validation_scores[0]
        score = float(np.mean(validation_scores))
        if epoch > warmup_epochs:
            scheduler.step(score)
        epoch_loss = total_loss / train_y.size
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(epoch_loss),
                "train_ce_loss": float(epoch_ce / train_y.size),
                "train_consistency_loss": float(epoch_consistency / train_y.size),
                "validation_car_balanced_accuracy": clean_score,
                "validation_balanced_accuracy": score,
                "validation_consistency_loss": validation_consistency,
                "validation_prediction_disagreement": validation_disagreement,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"LP epoch {epoch:02d}/{epochs} | loss={epoch_loss:.4f} | "
            f"val_car={clean_score:.4f} | val_mean={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_consistency = validation_consistency
            best_disagreement = validation_disagreement
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs > patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("REVE token probe did not produce a checkpoint")
    model.load_state_dict(best_state)
    return TorchProbeResult(
        predictor,
        best_epoch,
        best_score,
        best_consistency,
        best_disagreement,
        tuple(history),
    )
