"""Variable-cardinality set probe for frozen REVE tokens."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from .torch_probe import TorchProbeResult, TorchTokenPredictor, configure_torch_determinism


FeatureViews = NDArray[np.floating] | Sequence[NDArray[np.floating]]


def _as_feature_views(features: FeatureViews, *, name: str) -> tuple[np.ndarray, ...]:
    """Normalize dense or variable-token multi-view features."""

    if isinstance(features, np.ndarray):
        if features.ndim == 3:
            views = (features,)
        elif features.ndim == 4:
            views = tuple(features[:, index] for index in range(features.shape[1]))
        else:
            raise ValueError(
                f"{name} REVE set features have invalid shape {features.shape}; "
                "expected 3-D, 4-D, or a sequence of 3-D arrays"
            )
    else:
        views = tuple(np.asarray(view) for view in features)
    if not views or any(view.ndim != 3 for view in views):
        raise ValueError(f"{name} must contain at least one 3-D REVE token array")
    trial_counts = {int(view.shape[0]) for view in views}
    embedding_dims = {int(view.shape[-1]) for view in views}
    if len(trial_counts) != 1:
        raise ValueError(f"{name} views must contain aligned trial counts")
    if len(embedding_dims) != 1:
        raise ValueError(f"{name} views must share one REVE embedding dimension")
    return views


def fit_reve_set_probe(
    train_x: FeatureViews,
    train_y: NDArray[np.integer],
    val_x: FeatureViews,
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
    objective: str = "car_only",
    consistency_weight: float = 0.0,
    consistency_view_weights: Sequence[float] | None = None,
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

    train_views = _as_feature_views(train_x, name="train")
    val_views = _as_feature_views(val_x, name="validation")
    if len(train_views) != len(val_views):
        raise ValueError("Train and validation must contain the same observation views")
    if train_views[0].shape[0] != np.asarray(train_y).size:
        raise ValueError("Train token trials and labels are not aligned")
    if val_views[0].shape[0] != np.asarray(val_y).size:
        raise ValueError("Validation token trials and labels are not aligned")
    if train_views[0].shape[-1] != val_views[0].shape[-1]:
        raise ValueError("Train and validation REVE embedding dimensions differ")
    objective = objective.casefold()
    allowed_objectives = {
        "car_only",
        "multi_view_ce",
        "rule_consistency",
        "operator_consistency",
    }
    if objective not in allowed_objectives:
        raise ValueError(
            f"Unknown set-probe objective {objective!r}; expected {sorted(allowed_objectives)}"
        )
    if objective != "car_only" and len(train_views) < 2:
        raise ValueError(f"{objective} requires at least two aligned observation views")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight must be non-negative")
    if consistency_view_weights is None:
        view_weights = np.asarray(
            [0.0, *([1.0] * (len(train_views) - 1))], dtype=np.float64
        )
    else:
        view_weights = np.asarray(consistency_view_weights, dtype=np.float64)
    if view_weights.shape != (len(train_views),):
        raise ValueError("Provide exactly one consistency weight per observation view")
    if not np.isfinite(view_weights).all() or (view_weights < 0.0).any():
        raise ValueError("Observation-view consistency weights must be finite and non-negative")
    if objective == "operator_consistency":
        if view_weights[0] != 0.0 or not np.any(view_weights[1:] > 0.0):
            raise ValueError(
                "Operator consistency needs zero teacher weight and a positive student weight"
            )
        if consistency_weight <= 0.0:
            raise ValueError("Operator consistency requires a positive consistency_weight")
    if n_queries < 1 or n_heads < 1 or ff_multiplier < 1:
        raise ValueError("n_queries, n_heads, and ff_multiplier must be positive")
    embed_dim = int(train_views[0].shape[-1])
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
    train_view_tensors = [
        torch.from_numpy(np.asarray(view)) for view in train_views
    ]
    train_loader = DataLoader(
        TensorDataset(
            torch.arange(train_views[0].shape[0], dtype=torch.long),
            torch.from_numpy(np.asarray(train_y)),
        ),
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
        for indices, labels in train_loader:
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if global_step < total_warmup_steps:
                ratio = (10 ** (global_step / total_warmup_steps) - 1) / 9
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * max(ratio, 1e-3)
            optimizer.zero_grad(set_to_none=True)
            logits = torch.stack(
                [
                    model(
                        tokens[indices].to(
                            device, dtype=torch.float32, non_blocking=True
                        )
                    )
                    for tokens in train_view_tensors
                ],
                dim=1,
            )
            if objective == "car_only":
                ce_loss = criterion(logits[:, 0], labels)
            else:
                repeated_labels = labels[:, None].expand(-1, logits.shape[1])
                ce_loss = criterion(
                    logits.flatten(0, 1), repeated_labels.reshape(-1)
                )

            consistency_loss = logits.new_zeros(())
            if objective == "rule_consistency":
                log_probability = torch.log_softmax(logits, dim=-1)
                probability = log_probability.exp()
                mean_probability = probability.mean(dim=1, keepdim=True).clamp_min(
                    1e-8
                )
                consistency_loss = (
                    probability * (log_probability - mean_probability.log())
                ).sum(dim=-1).mean()
            elif objective == "operator_consistency":
                teacher_log_probability = torch.log_softmax(
                    logits[:, 0].detach(), dim=-1
                )
                teacher_probability = teacher_log_probability.exp()
                weighted_terms = []
                active_weights = []
                for view_index, view_weight in enumerate(view_weights[1:], start=1):
                    if view_weight <= 0.0:
                        continue
                    student_log_probability = torch.log_softmax(
                        logits[:, view_index], dim=-1
                    )
                    weighted_terms.append(
                        float(view_weight)
                        * (
                            teacher_probability
                            * (teacher_log_probability - student_log_probability)
                        )
                        .sum(dim=-1)
                        .mean()
                    )
                    active_weights.append(float(view_weight))
                consistency_loss = torch.stack(weighted_terms).sum() / sum(
                    active_weights
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
            [predictor.predict_proba(view) for view in val_views], axis=1
        )
        validation_predictions = validation_probabilities.argmax(axis=-1)
        validation_scores = [
            float(
                balanced_accuracy_score(
                    val_y, validation_predictions[:, view_index]
                )
            )
            for view_index in range(len(val_views))
        ]
        clipped_probability = np.clip(validation_probabilities, 1e-8, 1.0)
        if objective == "operator_consistency":
            teacher = clipped_probability[:, 0]
            teacher_log = np.log(teacher)
            validation_terms = [
                float(view_weight)
                * np.mean(
                    np.sum(
                        teacher
                        * (
                            teacher_log
                            - np.log(clipped_probability[:, view_index])
                        ),
                        axis=-1,
                    )
                )
                for view_index, view_weight in enumerate(
                    view_weights[1:], start=1
                )
                if view_weight > 0.0
            ]
            validation_consistency = float(
                np.sum(validation_terms) / np.sum(view_weights[1:])
            )
        elif len(val_views) > 1:
            mean_probability = np.clip(
                clipped_probability.mean(axis=1, keepdims=True), 1e-8, 1.0
            )
            validation_consistency = float(
                np.mean(
                    np.sum(
                        clipped_probability
                        * (
                            np.log(clipped_probability)
                            - np.log(mean_probability)
                        ),
                        axis=-1,
                    )
                )
            )
        else:
            validation_consistency = 0.0
        validation_disagreement = float(
            np.mean(
                np.any(
                    validation_predictions != validation_predictions[:, :1],
                    axis=1,
                )
            )
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
                "train_consistency_loss": float(
                    epoch_consistency / train_y.size
                ),
                "validation_car_balanced_accuracy": clean_score,
                "validation_balanced_accuracy": score,
                "validation_consistency_loss": validation_consistency,
                "validation_prediction_disagreement": validation_disagreement,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"Set LP q={n_queries} epoch {epoch:02d}/{epochs} | "
            f"loss={epoch_loss:.4f} | val_car={clean_score:.4f} | "
            f"val_mean={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_consistency = validation_consistency
            best_disagreement = validation_disagreement
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
        best_consistency,
        best_disagreement,
        tuple(history),
    )
