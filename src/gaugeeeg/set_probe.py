"""Variable-cardinality set probe for frozen REVE tokens."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from .torch_probe import TorchProbeResult, TorchTokenPredictor, configure_torch_determinism


def fit_reve_set_probe(
    train_x: NDArray[np.floating],
    train_y: NDArray[np.integer],
    val_x: NDArray[np.floating],
    val_y: NDArray[np.integer],
    *,
    initial_query: NDArray[np.floating],
    n_classes: int,
    seed: int,
    device: str,
    n_queries: int = 8,
    n_heads: int = 8,
    ff_multiplier: int = 2,
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
    """Fit a pooling-by-multihead-attention probe on a variable token set.

    Unlike the original token probe, this head never flattens the encoder's
    input tokens. A fixed bank of learned queries attends to however many REVE
    tokens are present, so the fitted predictor can be evaluated on a native
    channel subset without zero filling.
    """
    try:
        import torch
        from sklearn.metrics import balanced_accuracy_score
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        message = 'Set-probe dependencies are missing. Install with: pip install -e ".[reve]"'
        raise RuntimeError(message) from exc

    if train_x.ndim != 3 or val_x.ndim != 3:
        raise ValueError("REVE set features must have shape (trials, tokens, embedding_dim)")
    if train_x.shape[-1] != val_x.shape[-1]:
        raise ValueError("Train and validation REVE embedding dimensions differ")
    if n_queries < 1 or n_heads < 1 or ff_multiplier < 1:
        raise ValueError("n_queries, n_heads, and ff_multiplier must be positive")
    embed_dim = int(train_x.shape[-1])
    if embed_dim % n_heads:
        raise ValueError(f"embedding dimension {embed_dim} must be divisible by n_heads={n_heads}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the REVE set probe but is unavailable")

    configure_torch_determinism(seed, strict=deterministic)

    class ReveSetHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base_query = torch.as_tensor(initial_query, dtype=torch.float32).reshape(-1)
            if base_query.numel() != embed_dim:
                raise ValueError("REVE pretrained query dimension does not match cached tokens")
            base_query = base_query.reshape(1, embed_dim)
            generator = torch.Generator().manual_seed(seed + 104729)
            noise = torch.randn(n_queries, embed_dim, generator=generator) * 0.02
            self.queries = nn.Parameter(base_query.repeat(n_queries, 1) + noise)
            self.attention = nn.MultiheadAttention(
                embed_dim,
                n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(embed_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, ff_multiplier * embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_multiplier * embed_dim, embed_dim),
            )
            self.output_norm = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(n_queries * embed_dim, n_classes)

        def forward(self, tokens):
            queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            # Request explicit weights to avoid fused SDPA kernels whose exact
            # deterministic behavior varies across PyTorch/CUDA releases.
            context, _ = self.attention(queries, tokens, tokens, need_weights=True)
            context = self.attention_norm(queries + context)
            context = self.output_norm(context + self.feed_forward(context))
            return self.linear(self.dropout(context.flatten(1)))

    model = ReveSetHead().to(device)
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

        predictions = predictor.predict(val_x)
        score = float(balanced_accuracy_score(val_y, predictions))
        if epoch > warmup_epochs:
            scheduler.step(score)
        epoch_loss = total_loss / train_y.size
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(epoch_loss),
                "train_ce_loss": float(epoch_loss),
                "train_consistency_loss": 0.0,
                "validation_car_balanced_accuracy": score,
                "validation_balanced_accuracy": score,
                "validation_consistency_loss": 0.0,
                "validation_prediction_disagreement": 0.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"Set LP q={n_queries} epoch {epoch:02d}/{epochs} | "
            f"loss={epoch_loss:.4f} | val_car={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs > patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("REVE set probe did not produce a checkpoint")
    model.load_state_dict(best_state)
    return TorchProbeResult(
        predictor,
        best_epoch,
        best_score,
        0.0,
        0.0,
        tuple(history),
    )
