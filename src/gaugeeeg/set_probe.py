"""Variable-cardinality set probe for frozen REVE tokens."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from .torch_probe import (
    TorchProbeResult,
    TorchTokenPredictor,
    configure_torch_determinism,
)

FeatureViews = NDArray[np.floating] | Sequence[NDArray[np.floating]]
AuxiliaryViews = NDArray[np.floating] | Sequence[NDArray[np.floating]]


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


def _as_auxiliary_views(
    features: AuxiliaryViews,
    *,
    name: str,
) -> tuple[np.ndarray, ...]:
    """Normalize dense or multi-view GQBA token arrays."""

    if isinstance(features, np.ndarray):
        if features.ndim == 3:
            views = (features,)
        elif features.ndim == 4:
            views = tuple(features[:, index] for index in range(features.shape[1]))
        else:
            raise ValueError(
                f"{name} auxiliary features have invalid shape {features.shape}; "
                "expected 3-D, 4-D, or a sequence of 3-D arrays"
            )
    else:
        views = tuple(np.asarray(view) for view in features)
    if not views or any(view.ndim != 3 for view in views):
        raise ValueError(f"{name} must contain at least one 3-D auxiliary token array")
    if len({int(view.shape[0]) for view in views}) != 1:
        raise ValueError(f"{name} auxiliary views must contain aligned trial counts")
    if len({tuple(view.shape[1:]) for view in views}) != 1:
        raise ValueError(f"{name} auxiliary views must share token and feature dimensions")
    if any(not np.isfinite(view).all() for view in views):
        raise ValueError(f"{name} auxiliary features must be finite")
    return views


class TorchSetPredictor(TorchTokenPredictor):
    """Prediction adapter accepting REVE tokens or ``(REVE, auxiliary)``."""

    def _logits(self, x) -> NDArray[np.float32]:
        import torch

        if isinstance(x, tuple):
            if len(x) != 2:
                raise ValueError("Dual-input prediction expects (REVE tokens, auxiliary tokens)")
            tokens, auxiliary = (np.asarray(value) for value in x)
            if tokens.shape[0] != auxiliary.shape[0]:
                raise ValueError("REVE and auxiliary prediction trials are not aligned")
        else:
            tokens = np.asarray(x)
            auxiliary = None
        self.module.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, tokens.shape[0], self.batch_size):
                token_batch = torch.from_numpy(tokens[start : start + self.batch_size]).to(
                    self.device, dtype=torch.float32
                )
                auxiliary_batch = None
                if auxiliary is not None:
                    auxiliary_batch = torch.from_numpy(auxiliary[start : start + self.batch_size]).to(
                        self.device, dtype=torch.float32
                    )
                chunks.append(self.module(token_batch, auxiliary_batch).float().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def predict_auxiliary_components(self, x) -> dict[str, NDArray[np.float32]]:
        """Return final/base/residual logits and gates for adapter diagnostics."""

        import torch

        if not isinstance(x, tuple) or len(x) != 2:
            raise ValueError("Auxiliary diagnostics require (REVE tokens, auxiliary tokens)")
        tokens, auxiliary = (np.asarray(value) for value in x)
        if tokens.shape[0] != auxiliary.shape[0]:
            raise ValueError("REVE and auxiliary diagnostic trials are not aligned")
        collected = {name: [] for name in ("final", "base", "residual", "gate")}
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, tokens.shape[0], self.batch_size):
                token_batch = torch.from_numpy(tokens[start : start + self.batch_size]).to(
                    self.device, dtype=torch.float32
                )
                auxiliary_batch = torch.from_numpy(auxiliary[start : start + self.batch_size]).to(
                    self.device, dtype=torch.float32
                )
                output = self.module(token_batch, auxiliary_batch, return_components=True)
                for name, value in zip(collected, output, strict=True):
                    collected[name].append(value.float().cpu().numpy())
        return {
            name: np.concatenate(values, axis=0).astype(np.float32, copy=False)
            for name, values in collected.items()
        }


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
    train_auxiliary: AuxiliaryViews | None = None,
    val_auxiliary: AuxiliaryViews | None = None,
    auxiliary_queries: int = 2,
    auxiliary_hidden_dim: int = 64,
    auxiliary_fusion: str = "residual",
    auxiliary_gate_initial_probability: float = 0.25,
    auxiliary_preservation_weight: float = 0.0,
    auxiliary_residual_consistency_weight: float = 0.0,
    auxiliary_gate_supervision_weight: float = 0.0,
    auxiliary_target_classes: Sequence[int] = (2, 3),
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
    use_auxiliary = train_auxiliary is not None or val_auxiliary is not None
    if use_auxiliary and (train_auxiliary is None or val_auxiliary is None):
        raise ValueError("Provide both train_auxiliary and val_auxiliary")
    if use_auxiliary:
        train_aux_views = _as_auxiliary_views(train_auxiliary, name="train")
        val_aux_views = _as_auxiliary_views(val_auxiliary, name="validation")
        if len(train_aux_views) != len(train_views) or len(val_aux_views) != len(val_views):
            raise ValueError("Provide one aligned auxiliary array per REVE observation view")
        if train_aux_views[0].shape[0] != train_views[0].shape[0]:
            raise ValueError("Training REVE and auxiliary trial counts differ")
        if val_aux_views[0].shape[0] != val_views[0].shape[0]:
            raise ValueError("Validation REVE and auxiliary trial counts differ")
        if train_aux_views[0].shape[1:] != val_aux_views[0].shape[1:]:
            raise ValueError("Train and validation auxiliary token shapes differ")
    else:
        train_aux_views = ()
        val_aux_views = ()
    objective = objective.casefold()
    allowed_objectives = {
        "car_only",
        "multi_view_ce",
        "rule_consistency",
        "operator_consistency",
    }
    if objective not in allowed_objectives:
        raise ValueError(f"Unknown set-probe objective {objective!r}; expected {sorted(allowed_objectives)}")
    if objective != "car_only" and len(train_views) < 2:
        raise ValueError(f"{objective} requires at least two aligned observation views")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight must be non-negative")
    auxiliary_fusion = str(auxiliary_fusion).casefold()
    if auxiliary_fusion not in {"residual", "gated_residual"}:
        raise ValueError("auxiliary_fusion must be 'residual' or 'gated_residual'")
    if not use_auxiliary and auxiliary_fusion != "residual":
        raise ValueError("gated_residual requires auxiliary features")
    if not 0.0 < auxiliary_gate_initial_probability < 1.0:
        raise ValueError("auxiliary_gate_initial_probability must be in (0, 1)")
    if (
        auxiliary_preservation_weight < 0.0
        or auxiliary_residual_consistency_weight < 0.0
        or auxiliary_gate_supervision_weight < 0.0
    ):
        raise ValueError("Auxiliary loss weights must be non-negative")
    target_classes = tuple(sorted({int(value) for value in auxiliary_target_classes}))
    if any(value < 0 or value >= n_classes for value in target_classes):
        raise ValueError("auxiliary_target_classes must be valid class indices")
    if auxiliary_preservation_weight > 0.0 and not target_classes:
        raise ValueError("Class preservation requires at least one auxiliary target class")
    if not use_auxiliary and any(
        weight > 0.0
        for weight in (
            auxiliary_preservation_weight,
            auxiliary_residual_consistency_weight,
            auxiliary_gate_supervision_weight,
        )
    ):
        raise ValueError("Auxiliary losses require auxiliary features")
    if auxiliary_gate_supervision_weight > 0.0 and auxiliary_fusion != "gated_residual":
        raise ValueError("Gate supervision requires gated_residual fusion")
    if consistency_view_weights is None or objective == "car_only":
        # The car_only control trains on a single view and applies no
        # consistency term, so any globally configured multi-view weights are
        # irrelevant to it. Derive matching per-view weights from train_views
        # instead of failing the length check when the config carries weights
        # sized for the multi-montage arms.
        view_weights = np.asarray([0.0, *([1.0] * (len(train_views) - 1))], dtype=np.float64)
    else:
        view_weights = np.asarray(consistency_view_weights, dtype=np.float64)
    if view_weights.shape != (len(train_views),):
        raise ValueError("Provide exactly one consistency weight per observation view")
    if not np.isfinite(view_weights).all() or (view_weights < 0.0).any():
        raise ValueError("Observation-view consistency weights must be finite and non-negative")
    if objective == "operator_consistency":
        if view_weights[0] != 0.0 or not np.any(view_weights[1:] > 0.0):
            raise ValueError("Operator consistency needs zero teacher weight and a positive student weight")
        if consistency_weight <= 0.0:
            raise ValueError("Operator consistency requires a positive consistency_weight")
    if n_queries < 1 or n_heads < 1 or ff_multiplier < 1 or auxiliary_queries < 1 or auxiliary_hidden_dim < 1:
        raise ValueError("Probe query, head, feed-forward, and auxiliary sizes must be positive")
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

            self.uses_auxiliary = use_auxiliary
            if use_auxiliary:
                auxiliary_dim = int(train_aux_views[0].shape[-1])
                self.register_buffer(
                    "auxiliary_mean",
                    torch.as_tensor(
                        np.concatenate(train_aux_views, axis=0).mean(axis=(0, 1)),
                        dtype=torch.float32,
                    ).reshape(1, 1, auxiliary_dim),
                )
                auxiliary_scale = np.concatenate(train_aux_views, axis=0).std(axis=(0, 1))
                auxiliary_scale[auxiliary_scale < 1e-6] = 1.0
                self.register_buffer(
                    "auxiliary_scale",
                    torch.as_tensor(auxiliary_scale, dtype=torch.float32).reshape(1, 1, auxiliary_dim),
                )
                self.auxiliary_encoder = nn.Sequential(
                    nn.Linear(auxiliary_dim, auxiliary_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(auxiliary_hidden_dim),
                    nn.Linear(auxiliary_hidden_dim, embed_dim),
                )
                auxiliary_generator = torch.Generator().manual_seed(seed + 130363)
                self.auxiliary_queries = nn.Parameter(
                    torch.randn(
                        auxiliary_queries,
                        embed_dim,
                        generator=auxiliary_generator,
                    )
                    * 0.02
                )
                self.auxiliary_attention = nn.MultiheadAttention(
                    embed_dim,
                    n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.auxiliary_norm = nn.LayerNorm(embed_dim)
                self.auxiliary_linear = nn.Linear(auxiliary_queries * embed_dim, n_classes)
                # The first forward pass is exactly the existing REVE head.
                # The residual branch receives gradients through this layer,
                # then learns only when the data support an auxiliary update.
                nn.init.zeros_(self.auxiliary_linear.weight)
                nn.init.zeros_(self.auxiliary_linear.bias)
                if auxiliary_fusion == "gated_residual":
                    gate_features = auxiliary_queries * embed_dim + n_classes
                    self.auxiliary_gate = nn.Linear(gate_features, 1)
                    nn.init.zeros_(self.auxiliary_gate.weight)
                    initial_logit = math.log(
                        auxiliary_gate_initial_probability / (1.0 - auxiliary_gate_initial_probability)
                    )
                    nn.init.constant_(self.auxiliary_gate.bias, initial_logit)

        def forward(self, tokens, auxiliary=None, *, return_components=False):
            queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            # Request explicit weights to avoid fused SDPA kernels whose exact
            # deterministic behavior varies across PyTorch/CUDA releases.
            context, _ = self.attention(queries, tokens, tokens, need_weights=True)
            context = self.attention_norm(queries + context)
            context = self.output_norm(context + self.feed_forward(context))
            logits = self.linear(self.dropout(context.flatten(1)))
            if not self.uses_auxiliary:
                if auxiliary is not None:
                    raise ValueError("This REVE set head was fitted without auxiliary tokens")
                if return_components:
                    raise ValueError("Auxiliary components require an auxiliary branch")
                return logits
            if auxiliary is None:
                raise ValueError("This REVE set head requires auxiliary tokens")
            normalized = (auxiliary - self.auxiliary_mean) / self.auxiliary_scale
            auxiliary_tokens = self.auxiliary_encoder(normalized)
            auxiliary_queries_batch = self.auxiliary_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            auxiliary_context, _ = self.auxiliary_attention(
                auxiliary_queries_batch,
                auxiliary_tokens,
                auxiliary_tokens,
                need_weights=True,
            )
            auxiliary_context = self.auxiliary_norm(auxiliary_queries_batch + auxiliary_context)
            auxiliary_flat = self.dropout(auxiliary_context.flatten(1))
            residual = self.auxiliary_linear(auxiliary_flat)
            if auxiliary_fusion == "gated_residual":
                gate_input = torch.cat([auxiliary_flat, logits.detach()], dim=-1)
                gate = torch.sigmoid(self.auxiliary_gate(gate_input))
            else:
                gate = torch.ones((tokens.shape[0], 1), dtype=logits.dtype, device=logits.device)
            applied_residual = gate * residual
            final = logits + applied_residual
            if return_components:
                return final, logits, applied_residual, gate
            return final

    model = ReveSetHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_view_tensors = [torch.from_numpy(np.asarray(view)) for view in train_views]
    train_auxiliary_tensors = [torch.from_numpy(np.asarray(view)) for view in train_aux_views]
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
    predictor = TorchSetPredictor(model, device=device, batch_size=batch_size)
    total_warmup_steps = warmup_epochs * len(train_loader)
    global_step = 0
    best_score = -1.0
    best_epoch = 0
    best_consistency = float("nan")
    best_disagreement = float("nan")
    best_auxiliary_preservation = float("nan")
    best_auxiliary_consistency = float("nan")
    best_gate_target = float("nan")
    best_gate_nontarget = float("nan")
    best_gate_supervision = float("nan")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_ce = 0.0
        epoch_consistency = 0.0
        epoch_auxiliary_preservation = 0.0
        epoch_auxiliary_residual_consistency = 0.0
        epoch_auxiliary_gate_supervision = 0.0
        for indices, labels in train_loader:
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if global_step < total_warmup_steps:
                ratio = (10 ** (global_step / total_warmup_steps) - 1) / 9
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * max(ratio, 1e-3)
            optimizer.zero_grad(set_to_none=True)
            model_outputs = [
                model(
                    tokens[indices].to(device, dtype=torch.float32, non_blocking=True),
                    None
                    if not use_auxiliary
                    else train_auxiliary_tensors[view_index][indices].to(
                        device, dtype=torch.float32, non_blocking=True
                    ),
                    return_components=use_auxiliary,
                )
                for view_index, tokens in enumerate(train_view_tensors)
            ]
            if use_auxiliary:
                logits = torch.stack([output[0] for output in model_outputs], dim=1)
                base_logits = torch.stack([output[1] for output in model_outputs], dim=1)
                applied_residuals = torch.stack([output[2] for output in model_outputs], dim=1)
                auxiliary_gates = torch.stack([output[3] for output in model_outputs], dim=1)
            else:
                logits = torch.stack(model_outputs, dim=1)
                base_logits = None
                applied_residuals = None
                auxiliary_gates = None
            if objective == "car_only":
                ce_loss = criterion(logits[:, 0], labels)
            else:
                repeated_labels = labels[:, None].expand(-1, logits.shape[1])
                ce_loss = criterion(logits.flatten(0, 1), repeated_labels.reshape(-1))

            consistency_loss = logits.new_zeros(())
            if objective == "rule_consistency":
                log_probability = torch.log_softmax(logits, dim=-1)
                probability = log_probability.exp()
                mean_probability = probability.mean(dim=1, keepdim=True).clamp_min(1e-8)
                consistency_loss = (
                    (probability * (log_probability - mean_probability.log())).sum(dim=-1).mean()
                )
            elif objective == "operator_consistency":
                teacher_log_probability = torch.log_softmax(logits[:, 0].detach(), dim=-1)
                teacher_probability = teacher_log_probability.exp()
                weighted_terms = []
                active_weights = []
                for view_index, view_weight in enumerate(view_weights[1:], start=1):
                    if view_weight <= 0.0:
                        continue
                    student_log_probability = torch.log_softmax(logits[:, view_index], dim=-1)
                    weighted_terms.append(
                        float(view_weight)
                        * (teacher_probability * (teacher_log_probability - student_log_probability))
                        .sum(dim=-1)
                        .mean()
                    )
                    active_weights.append(float(view_weight))
                consistency_loss = torch.stack(weighted_terms).sum() / sum(active_weights)

            auxiliary_preservation_loss = logits.new_zeros(())
            auxiliary_residual_consistency_loss = logits.new_zeros(())
            auxiliary_gate_supervision_loss = logits.new_zeros(())
            target_mask = torch.zeros_like(labels, dtype=torch.bool)
            for class_index in target_classes:
                target_mask |= labels.eq(class_index)
            if use_auxiliary and auxiliary_preservation_weight > 0.0:
                preserve_mask = ~target_mask
                if torch.any(preserve_mask):
                    # The safeguard may update only the auxiliary correction;
                    # ordinary CE remains responsible for the REVE base head.
                    teacher_probability = torch.softmax(base_logits.detach(), dim=-1)
                    preservation_logits = base_logits.detach() + applied_residuals
                    student_log_probability = torch.log_softmax(preservation_logits, dim=-1)
                    per_view_kl = (
                        teacher_probability
                        * (torch.log(teacher_probability.clamp_min(1e-8)) - student_log_probability)
                    ).sum(dim=-1)
                    auxiliary_preservation_loss = per_view_kl[preserve_mask].mean()
            if (
                use_auxiliary
                and auxiliary_residual_consistency_weight > 0.0
                and applied_residuals.shape[1] > 1
            ):
                auxiliary_residual_consistency_loss = (
                    (applied_residuals[:, 1:] - applied_residuals[:, :1]).pow(2).mean()
                )
            if use_auxiliary and auxiliary_gate_supervision_weight > 0.0:
                gate_target = target_mask[:, None, None].expand_as(auxiliary_gates)
                auxiliary_gate_supervision_loss = torch.nn.functional.binary_cross_entropy(
                    auxiliary_gates, gate_target.to(dtype=auxiliary_gates.dtype)
                )
            loss = (
                ce_loss
                + consistency_weight * consistency_loss
                + auxiliary_preservation_weight * auxiliary_preservation_loss
                + auxiliary_residual_consistency_weight * auxiliary_residual_consistency_loss
                + auxiliary_gate_supervision_weight * auxiliary_gate_supervision_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
            total_loss += float(loss.item()) * labels.shape[0]
            epoch_ce += float(ce_loss.item()) * labels.shape[0]
            epoch_consistency += float(consistency_loss.item()) * labels.shape[0]
            epoch_auxiliary_preservation += float(auxiliary_preservation_loss.item()) * labels.shape[0]
            epoch_auxiliary_residual_consistency += (
                float(auxiliary_residual_consistency_loss.item()) * labels.shape[0]
            )
            epoch_auxiliary_gate_supervision += (
                float(auxiliary_gate_supervision_loss.item()) * labels.shape[0]
            )
            global_step += 1

        validation_probabilities = np.stack(
            [
                predictor.predict_proba(view if not use_auxiliary else (view, val_aux_views[view_index]))
                for view_index, view in enumerate(val_views)
            ],
            axis=1,
        )
        validation_predictions = validation_probabilities.argmax(axis=-1)
        validation_scores = [
            float(balanced_accuracy_score(val_y, validation_predictions[:, view_index]))
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
                        teacher * (teacher_log - np.log(clipped_probability[:, view_index])),
                        axis=-1,
                    )
                )
                for view_index, view_weight in enumerate(view_weights[1:], start=1)
                if view_weight > 0.0
            ]
            validation_consistency = float(np.sum(validation_terms) / np.sum(view_weights[1:]))
        elif len(val_views) > 1:
            mean_probability = np.clip(clipped_probability.mean(axis=1, keepdims=True), 1e-8, 1.0)
            validation_consistency = float(
                np.mean(
                    np.sum(
                        clipped_probability * (np.log(clipped_probability) - np.log(mean_probability)),
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
        validation_auxiliary_preservation = 0.0
        validation_auxiliary_consistency = 0.0
        validation_gate_target = float("nan")
        validation_gate_nontarget = float("nan")
        validation_gate_supervision = 0.0
        if use_auxiliary:
            components = [
                predictor.predict_auxiliary_components((view, val_aux_views[view_index]))
                for view_index, view in enumerate(val_views)
            ]
            final_logits = np.stack([item["final"] for item in components], axis=1)
            base_logits_numpy = np.stack([item["base"] for item in components], axis=1)
            residuals_numpy = np.stack([item["residual"] for item in components], axis=1)
            gates_numpy = np.stack([item["gate"] for item in components], axis=1)
            target_mask_numpy = np.isin(val_y, target_classes)
            preserve_mask_numpy = ~target_mask_numpy
            if np.any(preserve_mask_numpy):
                teacher = base_logits_numpy - base_logits_numpy.max(axis=-1, keepdims=True)
                teacher = np.exp(teacher)
                teacher /= teacher.sum(axis=-1, keepdims=True)
                student = final_logits - final_logits.max(axis=-1, keepdims=True)
                student = np.exp(student)
                student /= student.sum(axis=-1, keepdims=True)
                validation_auxiliary_preservation = float(
                    np.mean(
                        np.sum(
                            teacher[preserve_mask_numpy]
                            * (
                                np.log(np.clip(teacher[preserve_mask_numpy], 1e-8, 1.0))
                                - np.log(np.clip(student[preserve_mask_numpy], 1e-8, 1.0))
                            ),
                            axis=-1,
                        )
                    )
                )
            if residuals_numpy.shape[1] > 1:
                validation_auxiliary_consistency = float(
                    np.mean((residuals_numpy[:, 1:] - residuals_numpy[:, :1]) ** 2)
                )
            if np.any(target_mask_numpy):
                validation_gate_target = float(np.mean(gates_numpy[target_mask_numpy]))
            if np.any(preserve_mask_numpy):
                validation_gate_nontarget = float(np.mean(gates_numpy[preserve_mask_numpy]))
            gate_targets_numpy = np.broadcast_to(target_mask_numpy[:, None, None], gates_numpy.shape).astype(
                np.float64
            )
            clipped_gates = np.clip(gates_numpy.astype(np.float64), 1e-8, 1.0 - 1e-8)
            validation_gate_supervision = float(
                -np.mean(
                    gate_targets_numpy * np.log(clipped_gates)
                    + (1.0 - gate_targets_numpy) * np.log(1.0 - clipped_gates)
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
                "train_consistency_loss": float(epoch_consistency / train_y.size),
                "train_auxiliary_preservation_loss": float(epoch_auxiliary_preservation / train_y.size),
                "train_auxiliary_residual_consistency_loss": float(
                    epoch_auxiliary_residual_consistency / train_y.size
                ),
                "train_auxiliary_gate_supervision_loss": float(
                    epoch_auxiliary_gate_supervision / train_y.size
                ),
                "validation_auxiliary_preservation_loss": validation_auxiliary_preservation,
                "validation_auxiliary_consistency_loss": validation_auxiliary_consistency,
                "validation_auxiliary_gate_target_mean": validation_gate_target,
                "validation_auxiliary_gate_nontarget_mean": validation_gate_nontarget,
                "validation_auxiliary_gate_supervision_loss": validation_gate_supervision,
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
            best_auxiliary_preservation = validation_auxiliary_preservation
            best_auxiliary_consistency = validation_auxiliary_consistency
            best_gate_target = validation_gate_target
            best_gate_nontarget = validation_gate_nontarget
            best_gate_supervision = validation_gate_supervision
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
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    auxiliary_parameters = 0
    if use_auxiliary:
        auxiliary_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("auxiliary_") and parameter.requires_grad
        )
    return TorchProbeResult(
        predictor,
        best_epoch,
        best_score,
        best_consistency,
        best_disagreement,
        tuple(history),
        trainable_parameters,
        auxiliary_parameters,
        best_auxiliary_preservation,
        best_auxiliary_consistency,
        best_gate_target,
        best_gate_nontarget,
        best_gate_supervision,
    )
