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
    history: tuple[dict[str, float], ...]


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
) -> TorchProbeResult:
    """Fit REVE's released token-level LP head with a frozen cached encoder."""
    try:
        import torch
        from sklearn.metrics import balanced_accuracy_score
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        message = 'Token probe dependencies are missing. Install with: pip install -e ".[reve]"'
        raise RuntimeError(message) from exc

    if train_x.ndim != 3 or val_x.ndim != 3:
        raise ValueError("reve_token probe expects features shaped (trials, tokens, embedding_dim)")
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
            n_tokens, embed_dim = train_x.shape[1:]
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
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for tokens, labels in train_loader:
            tokens = tokens.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if global_step < total_warmup_steps:
                ratio = (10 ** (global_step / total_warmup_steps) - 1) / 9
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * max(ratio, 1e-3)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
            total_loss += float(loss.item()) * labels.shape[0]
            global_step += 1

        prediction = predictor.predict(val_x)
        score = float(balanced_accuracy_score(val_y, prediction))
        if epoch > warmup_epochs:
            scheduler.step(score)
        epoch_loss = total_loss / train_y.size
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(epoch_loss),
                "validation_balanced_accuracy": score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(f"LP epoch {epoch:02d}/{epochs} | loss={epoch_loss:.4f} | val_bacc={score:.4f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
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
    return TorchProbeResult(predictor, best_epoch, best_score, tuple(history))
