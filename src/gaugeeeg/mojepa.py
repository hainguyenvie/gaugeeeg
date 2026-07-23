"""Gauge-aware measurement-operator predictive adaptation for REVE.

The module deliberately keeps the adaptation head separate from the released
encoder.  Only low-rank attention updates are saved; the predictor/classifier
is a training-time pretext head and is discarded before downstream probing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import dump_config
from .datasets import EEGDataset, load_physionet_mi
from .montage import montage_keep_mask, parse_observation_view, prepare_observation_view

OBJECTIVES = ("lora_multiview_ce", "lora_generic_jepa", "gauge_mojepa")
REFERENCE_ORDER = ("car", "cz", "pz", "fz")
OPERATOR_BASE_DIM = 9


def measurement_operator_code(
    view: str,
    channel_names: Sequence[str],
) -> np.ndarray:
    """Encode the known measurement operator without using task labels.

    The code exposes reference identity, acquisition policy, retained-channel
    fraction, lossiness and the exact binary sensor mask in canonical order.
    It intentionally contains no class, subject or dataset identifier.
    """

    specification = parse_observation_view(view)
    reference = specification.reference.casefold()
    if reference not in REFERENCE_ORDER:
        raise ValueError(f"Gauge-MOJEPA supports references {REFERENCE_ORDER}, received {reference!r}")
    mask = montage_keep_mask(channel_names, specification.montage)
    code = np.zeros(OPERATOR_BASE_DIM + len(channel_names), dtype=np.float32)
    code[REFERENCE_ORDER.index(reference)] = 1.0
    code[4] = float(specification.channel_policy == "full")
    code[5] = float(specification.channel_policy == "remove")
    retained = float(mask.mean())
    code[6] = retained
    code[7] = float(retained < 1.0)
    code[8] = float(reference != "car")
    code[OPERATOR_BASE_DIM:] = mask.astype(np.float32)
    return code


def gauge_reference_groups(views: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Return same-montage groups eligible for exact gauge consistency."""

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, view in enumerate(views):
        specification = parse_observation_view(view)
        key = (specification.montage, specification.channel_policy)
        grouped.setdefault(key, []).append(index)
    return tuple(tuple(indices) for indices in grouped.values() if len(indices) > 1)


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.nn.utils import parametrize
    except ImportError as exc:
        raise RuntimeError('Gauge-MOJEPA requires: pip install -e ".[data,reve]"') from exc
    return torch, nn, parametrize


def inject_reve_lora(
    model: Any,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    last_n_blocks: int = 4,
) -> list[str]:
    """Freeze ``model`` and attach LoRA to its last attention blocks.

    Both split Q/K/V/O projections and PyTorch's fused QKV MultiheadAttention
    layout are supported.  The latter receives one low-rank update on the
    fused ``in_proj_weight`` plus one on the output projection.
    """

    torch, nn, parametrize = _require_torch()
    if rank < 1 or last_n_blocks < 1 or not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("LoRA rank, alpha and last_n_blocks must be positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class _LoRAWeight(nn.Module):
        def __init__(self, shape: tuple[int, int]) -> None:
            super().__init__()
            out_features, in_features = shape
            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))
            self.scaling = float(alpha) / float(rank)
            self.enabled = True

        def forward(self, weight):
            if not self.enabled:
                return weight
            return weight + (self.lora_b @ self.lora_a).to(weight.dtype) * self.scaling

    explicit_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "query",
        "key",
        "value",
        "to_q",
        "to_k",
        "to_v",
        "to_qkv",
        "to_out",
    }
    candidates: list[tuple[str, str, Any, str]] = []
    mha_paths: list[str] = []
    for path, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            mha_paths.append(path)
            if module.in_proj_weight is not None:
                candidates.append((path, f"{path}.in_proj_weight", module, "in_proj_weight"))
            candidates.append((path, f"{path}.out_proj.weight", module.out_proj, "weight"))

    for path, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not path:
            continue
        leaf = path.rsplit(".", maxsplit=1)[-1].casefold()
        if leaf not in explicit_names:
            continue
        if any(path == f"{mha_path}.out_proj" for mha_path in mha_paths):
            continue
        owner = path.rsplit(".", maxsplit=1)[0]
        candidates.append((owner, f"{path}.weight", module, "weight"))

    if not candidates:
        raise RuntimeError(
            "No REVE attention projections were found. Expected split Q/K/V/O linear layers "
            "or torch.nn.MultiheadAttention modules."
        )
    owners = list(dict.fromkeys(owner for owner, _, _, _ in candidates))
    selected_owners = set(owners[-last_n_blocks:])
    selected = [candidate for candidate in candidates if candidate[0] in selected_owners]
    attached: list[str] = []
    seen: set[tuple[int, str]] = set()
    for _, parameter_path, module, parameter_name in selected:
        identity = (id(module), parameter_name)
        if identity in seen:
            continue
        seen.add(identity)
        weight = getattr(module, parameter_name)
        if weight.ndim != 2:
            raise RuntimeError(f"LoRA target {parameter_path} is not a matrix")
        parametrize.register_parametrization(
            module,
            parameter_name,
            _LoRAWeight(tuple(int(value) for value in weight.shape)),
        )
        getattr(module.parametrizations, parameter_name).original.requires_grad_(False)
        attached.append(parameter_path)
    if not attached:
        raise RuntimeError("No LoRA parametrizations were attached")
    return attached


def set_lora_enabled(model: Any, enabled: bool) -> None:
    """Enable student adapters or disable them for the frozen teacher path."""

    for module in model.modules():
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            module.enabled = bool(enabled)


def lora_state_dict(model: Any) -> dict[str, Any]:
    """Return only trainable low-rank tensors, never released REVE weights."""

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in trainable}


def lora_parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def load_mojepa_adapter(model: Any, checkpoint: str | Path, *, map_location: Any = "cpu") -> dict:
    """Inject the recorded LoRA layout and load a Gauge-MOJEPA adapter."""

    torch, _, _ = _require_torch()
    payload = torch.load(Path(checkpoint), map_location=map_location, weights_only=True)
    metadata = payload["metadata"]
    expected_revision = str(metadata.get("model_revision", "unresolved"))
    current_revision = str(
        getattr(getattr(model, "config", None), "_commit_hash", None) or "unresolved"
    )
    if (
        expected_revision != "unresolved"
        and current_revision != "unresolved"
        and expected_revision != current_revision
    ):
        raise RuntimeError(
            f"Adapter expects REVE revision {expected_revision}, loaded {current_revision}"
        )
    attached = inject_reve_lora(
        model,
        rank=int(metadata["rank"]),
        alpha=float(metadata["alpha"]),
        last_n_blocks=int(metadata["last_n_blocks"]),
    )
    if attached != list(metadata["targets"]):
        raise RuntimeError(
            "Adapter target layout differs from the checkpoint; pin the recorded REVE revision"
        )
    incompatible = model.load_state_dict(payload["adapter_state"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_adapter = [name for name in incompatible.missing_keys if ".lora_a" in name or ".lora_b" in name]
    if unexpected or missing_adapter:
        raise RuntimeError(
            f"Could not load adapter (unexpected={unexpected}, missing_adapter={missing_adapter})"
        )
    set_lora_enabled(model, True)
    return dict(metadata)


def _pooled_embedding(model: Any, tokens: Any):
    torch, _, _ = _require_torch()
    if hasattr(model, "attention_pooling"):
        pooled = model.attention_pooling(tokens)
        if pooled.ndim > 2:
            pooled = pooled.reshape(pooled.shape[0], -1, pooled.shape[-1]).mean(dim=1)
    else:
        pooled = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1]).mean(dim=1)
    return torch.nn.functional.normalize(pooled.float(), dim=-1)


def _position_tensor(position_bank: Any, channel_names: Sequence[str], device: Any):
    torch, _, _ = _require_torch()
    with torch.inference_mode():
        positions = position_bank(list(channel_names)).float().to(device)
    if positions.shape[0] != len(channel_names):
        raise ValueError(
            f"Position bank returned {positions.shape[0]} rows for {len(channel_names)} channels"
        )
    return positions


def _encode_numpy_view(
    model: Any,
    position_bank: Any,
    x_uv: np.ndarray,
    channel_names: Sequence[str],
    view: str,
    *,
    device: Any,
    input_scale_uv: float,
    reference_seed: int,
):
    torch, _, _ = _require_torch()
    observed, observed_names = prepare_observation_view(
        x_uv,
        channel_names,
        view,
        seed=reference_seed,
    )
    signal = torch.from_numpy(np.asarray(observed, dtype=np.float32) / input_scale_uv).to(device)
    positions = _position_tensor(position_bank, observed_names, device)
    batch_positions = positions.unsqueeze(0).expand(signal.shape[0], -1, -1)
    return _pooled_embedding(model, model(signal, batch_positions))


def _cosine_loss(left: Any, right: Any):
    torch, _, _ = _require_torch()
    similarity = torch.nn.functional.cosine_similarity(left, right, dim=-1).clamp(-1.0, 1.0)
    return (1.0 - similarity).mean()


def _limit_indices(size: int, maximum: int | None, rng: np.random.Generator) -> np.ndarray:
    indices = np.arange(size, dtype=np.int64)
    if maximum is not None and int(maximum) < 1:
        raise ValueError("Trial caps must be positive or null")
    if maximum is not None and int(maximum) < size:
        indices = np.sort(rng.choice(indices, size=int(maximum), replace=False))
    return indices


def _batch_indices(indices: np.ndarray, batch_size: int, *, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(indices) if shuffle else indices
    for start in range(0, order.size, batch_size):
        yield order[start : start + batch_size]


def run_mojepa_adaptation(
    config: dict[str, Any],
    *,
    objective: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Adapt REVE on the locked train split and select on the locked val split."""

    torch, nn, _ = _require_torch()
    from transformers import AutoModel

    adaptation = dict(config.get("adaptation", {}))
    experiment = config["experiment"]
    objective = str(objective or adaptation.get("objective", "gauge_mojepa")).casefold()
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}; expected {OBJECTIVES}")
    seed = int(seed if seed is not None else adaptation.get("seed", config.get("seed", 7)))
    requested_device = str(device or adaptation.get("device", experiment.get("device", "auto")))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(requested_device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    views = tuple(str(view) for view in adaptation.get("views", experiment.get("training_views", ["car"])))
    if not views or views[0].casefold() != "car":
        raise ValueError("Gauge-MOJEPA views must begin with full-montage CAR")
    channel_names: Sequence[str]
    dataset = load_physionet_mi(config["data"], force_recompute=False)
    if not np.isclose(dataset.sfreq, 200.0):
        raise ValueError("REVE adaptation requires 200 Hz input")
    channel_names = dataset.channel_names
    for view in views:
        measurement_operator_code(view, channel_names)

    try:
        position_bank = (
            AutoModel.from_pretrained(
                experiment.get("position_model_name", "brain-bzh/reve-positions"),
                revision=experiment.get("position_model_revision"),
                trust_remote_code=True,
            )
            .to(torch_device)
            .eval()
        )
        model = AutoModel.from_pretrained(
            experiment.get("model_name", "brain-bzh/reve-base"),
            revision=experiment.get("model_revision"),
            trust_remote_code=True,
        ).to(torch_device)
    except Exception as exc:
        raise RuntimeError(
            "Could not load gated REVE weights. Accept the Hugging Face agreement and run hf auth login."
        ) from exc
    for parameter in position_bank.parameters():
        parameter.requires_grad_(False)

    rank = int(adaptation.get("lora_rank", 8))
    alpha = float(adaptation.get("lora_alpha", 16.0))
    last_n_blocks = int(adaptation.get("lora_last_n_blocks", 4))
    targets = inject_reve_lora(model, rank=rank, alpha=alpha, last_n_blocks=last_n_blocks)
    adapter_parameters = lora_parameter_count(model)

    train = dataset.subset([int(value) for value in config["data"]["train_subjects"]])
    validation = dataset.subset([int(value) for value in config["data"]["val_subjects"]])
    rng = np.random.default_rng(seed)
    train_indices = _limit_indices(train.y.size, adaptation.get("max_train_trials"), rng)
    val_indices = _limit_indices(validation.y.size, adaptation.get("max_val_trials"), rng)
    batch_size = int(adaptation.get("batch_size", 8))
    epochs = int(adaptation.get("epochs", 5))
    patience = int(adaptation.get("patience", 2))
    learning_rate = float(adaptation.get("learning_rate", 2e-4))
    weight_decay = float(adaptation.get("weight_decay", 1e-2))
    clip_grad = float(adaptation.get("clip_grad", 1.0))
    prediction_weight = float(adaptation.get("prediction_weight", 1.0))
    gauge_weight = float(adaptation.get("gauge_weight", 0.25))
    anchor_weight = float(adaptation.get("anchor_weight", 0.1))
    input_scale_uv = float(experiment.get("reve_input_scale_uv", 100.0))
    reference_seed = int(experiment.get("reference_seed", 7))
    if batch_size < 1 or epochs < 1 or patience < 1:
        raise ValueError("batch_size, epochs and patience must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0 or clip_grad <= 0.0:
        raise ValueError("learning_rate/clip_grad must be positive and weight_decay non-negative")
    if any(value < 0.0 for value in (prediction_weight, gauge_weight, anchor_weight)):
        raise ValueError("Gauge-MOJEPA loss weights must be non-negative")
    operator_codes = torch.from_numpy(
        np.stack([measurement_operator_code(view, channel_names) for view in views])
    ).to(torch_device)
    operator_code_dim = int(operator_codes.shape[-1])
    gauge_groups = gauge_reference_groups(views)

    bootstrap_index = train_indices[: min(batch_size, train_indices.size)]
    model.eval()
    set_lora_enabled(model, True)
    with torch.inference_mode():
        bootstrap_embedding = _encode_numpy_view(
            model,
            position_bank,
            train.x_uv[bootstrap_index],
            channel_names,
            views[0],
            device=torch_device,
            input_scale_uv=input_scale_uv,
            reference_seed=reference_seed,
        )
    embedding_dim = int(bootstrap_embedding.shape[-1])
    hidden_dim = int(adaptation.get("predictor_hidden_dim", min(512, embedding_dim)))
    predictor = nn.Sequential(
        nn.Linear(embedding_dim + operator_code_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, embedding_dim),
    ).to(torch_device)
    classifier = nn.Linear(embedding_dim, len(dataset.label_names)).to(torch_device)
    head = classifier if objective == "lora_multiview_ce" else predictor
    optimizer = torch.optim.AdamW(
        [
            *[parameter for parameter in model.parameters() if parameter.requires_grad],
            *head.parameters(),
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def batch_loss(split: EEGDataset, batch: np.ndarray, *, training: bool):
        x_uv = split.x_uv[batch]
        labels = torch.from_numpy(split.y[batch]).long().to(torch_device)
        # Keep REVE in eval mode while retaining autograd. Its official forward
        # adds random coordinate noise in train mode, which would confound the
        # exact same-montage gauge term with a second stochastic intervention.
        model.eval()
        head.train(training)
        teacher = None
        if objective != "lora_multiview_ce":
            set_lora_enabled(model, False)
            model.eval()
            with torch.no_grad():
                teacher = _encode_numpy_view(
                    model,
                    position_bank,
                    x_uv,
                    channel_names,
                    "car",
                    device=torch_device,
                    input_scale_uv=input_scale_uv,
                    reference_seed=reference_seed,
                ).detach()
            model.eval()
        set_lora_enabled(model, True)
        students = [
            _encode_numpy_view(
                model,
                position_bank,
                x_uv,
                channel_names,
                view,
                device=torch_device,
                input_scale_uv=input_scale_uv,
                reference_seed=reference_seed,
            )
            for view in views
        ]
        zero = students[0].new_zeros(())
        if objective == "lora_multiview_ce":
            supervised = torch.stack(
                [torch.nn.functional.cross_entropy(classifier(value), labels) for value in students]
            ).mean()
            return supervised, {"prediction": zero, "gauge": zero, "anchor": zero, "ce": supervised}
        if teacher is None:
            raise RuntimeError("Predictive adaptation requires a frozen teacher embedding")

        prediction_losses = []
        for view_index, student in enumerate(students):
            code = operator_codes[view_index].expand(student.shape[0], -1)
            if objective == "lora_generic_jepa":
                code = torch.zeros_like(code)
            prediction_losses.append(_cosine_loss(predictor(torch.cat([student, code], dim=-1)), teacher))
        prediction = torch.stack(prediction_losses).mean()
        gauge = zero
        anchor = zero
        if objective == "gauge_mojepa":
            gauge_terms = []
            for group in gauge_groups:
                group_anchor = students[group[0]]
                gauge_terms.extend(_cosine_loss(group_anchor, students[index]) for index in group[1:])
            gauge = torch.stack(gauge_terms).mean() if gauge_terms else zero
            anchor = _cosine_loss(students[0], teacher)
        total = prediction_weight * prediction + gauge_weight * gauge + anchor_weight * anchor
        return total, {"prediction": prediction, "gauge": gauge, "anchor": anchor, "ce": zero}

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    for epoch in range(epochs):
        totals = {name: 0.0 for name in ("loss", "prediction", "gauge", "anchor", "ce")}
        batches = 0
        for batch in _batch_indices(train_indices, batch_size, shuffle=True, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            loss, components = batch_loss(train, batch, training=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], clip_grad
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in components.items():
                totals[name] += float(value.detach())
            batches += 1

        validation_total = 0.0
        validation_batches = 0
        with torch.no_grad():
            for batch in _batch_indices(val_indices, batch_size, shuffle=False, rng=rng):
                loss, _ = batch_loss(validation, batch, training=False)
                validation_total += float(loss)
                validation_batches += 1
        validation_loss = validation_total / max(validation_batches, 1)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / max(batches, 1),
            "validation_loss": validation_loss,
        }
        row.update(
            {f"train_{name}_loss": totals[name] / max(batches, 1) for name in totals if name != "loss"}
        )
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = lora_state_dict(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Adaptation did not produce a finite validation checkpoint")

    resolved_model_revision = str(
        getattr(model.config, "_commit_hash", None) or experiment.get("model_revision") or "unresolved"
    )
    resolved_position_revision = str(
        getattr(position_bank.config, "_commit_hash", None)
        or experiment.get("position_model_revision")
        or "unresolved"
    )
    output = Path(output_dir or adaptation.get("output_dir", f"outputs/reve_mojepa/{objective}_s{seed}"))
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "gaugeeeg-reve-lora-v1",
        "method": objective,
        "rank": rank,
        "alpha": alpha,
        "last_n_blocks": last_n_blocks,
        "targets": targets,
        "model_name": experiment.get("model_name", "brain-bzh/reve-base"),
        "model_revision": resolved_model_revision,
        "position_model_revision": resolved_position_revision,
        "seed": seed,
        "views": list(views),
        "adapter_parameters": adapter_parameters,
    }
    checkpoint = output / "adapter.pt"
    torch.save({"metadata": metadata, "adapter_state": best_state}, checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary = {
        **metadata,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "train_trials": int(train_indices.size),
        "validation_trials": int(val_indices.size),
        "uses_task_labels": objective == "lora_multiview_ce",
        "teacher": None if objective == "lora_multiview_ce" else "frozen_reve_full_car",
        "training_head_discarded": True,
        "reserved_test_subjects_used_for_adaptation_or_scoring": False,
        "loss_weights": {
            "prediction": prediction_weight if objective != "lora_multiview_ce" else 0.0,
            "gauge": gauge_weight if objective == "gauge_mojepa" else 0.0,
            "anchor": anchor_weight if objective == "gauge_mojepa" else 0.0,
        },
        "history": history,
    }
    (output / "adaptation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    dump_config(config, output / "resolved_config.yaml")
    return summary
