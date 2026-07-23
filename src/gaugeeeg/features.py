"""Feature encoders used by the pilot benchmark."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch


class Encoder(Protocol):
    name: str
    cache_signature: str
    metadata: dict[str, str | float]

    def transform(
        self,
        x_uv: NDArray[np.floating],
        channel_names: Sequence[str],
        sfreq: float,
    ) -> NDArray[np.float32]: ...


class BandpowerEncoder:
    """Log absolute bandpower per channel; intentionally reference-sensitive."""

    name = "bandpower"
    cache_signature = "bandpower:v1"
    metadata = {"encoder": "bandpower", "encoder_revision": "v1"}

    def __init__(self, bands: dict[str, tuple[float, float]] | None = None) -> None:
        self.bands = bands or {
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
        }

    def transform(
        self,
        x_uv: NDArray[np.floating],
        channel_names: Sequence[str],
        sfreq: float,
    ) -> NDArray[np.float32]:
        del channel_names
        x = np.asarray(x_uv, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"Expected (trials, channels, time), got {x.shape}")
        nperseg = min(x.shape[-1], max(32, int(round(2.0 * sfreq))))
        frequencies, psd = welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
        features = []
        for low, high in self.bands.values():
            mask = (frequencies >= low) & (frequencies < high)
            if mask.sum() < 2:
                raise ValueError(f"Insufficient FFT bins for band [{low}, {high})")
            power = np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)
            features.append(np.log(np.maximum(power, np.finfo(np.float64).tiny)))
        stacked = np.stack(features, axis=-1)
        return stacked.reshape(stacked.shape[0], -1).astype(np.float32)


class FrozenREVEEncoder:
    """Frozen REVE feature extractor using the authors' Hugging Face interface."""

    name = "reve"

    def __init__(
        self,
        *,
        model_name: str,
        position_model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        pooling: str = "attention",
        input_scale_uv: float = 100.0,
        model_revision: str | None = None,
        position_model_revision: str | None = None,
        adapter_checkpoint: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            message = 'REVE dependencies are missing. Install with: pip install -e ".[reve]"'
            raise RuntimeError(message) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if pooling not in {"attention", "mean", "tokens"}:
            raise ValueError("REVE pooling must be 'attention', 'mean', or 'tokens'")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.pooling = pooling
        self.input_scale_uv = float(input_scale_uv)
        self.model_name = model_name

        try:
            self.position_bank = AutoModel.from_pretrained(
                position_model_name,
                revision=position_model_revision,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                revision=model_revision,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load REVE. Accept the model agreement at "
                "https://huggingface.co/brain-bzh/reve-base and run `hf auth login`."
            ) from exc
        self.position_bank.to(self.device).eval()
        self.model.to(self.device).eval()
        adapter_metadata = None
        adapter_sha256 = None
        if adapter_checkpoint:
            from .mojepa import load_mojepa_adapter

            adapter_path = Path(adapter_checkpoint)
            if not adapter_path.is_file():
                raise FileNotFoundError(f"REVE adapter checkpoint does not exist: {adapter_path}")
            adapter_metadata = load_mojepa_adapter(
                self.model,
                adapter_path,
                map_location=self.device,
            )
            adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
            self.model.eval()
        resolved_model_revision = str(
            getattr(self.model.config, "_commit_hash", None) or model_revision or "unresolved"
        )
        resolved_position_revision = str(
            getattr(self.position_bank.config, "_commit_hash", None)
            or position_model_revision
            or "unresolved"
        )
        self.metadata = {
            "encoder": "reve",
            "model_name": model_name,
            "model_revision": resolved_model_revision,
            "position_model_name": position_model_name,
            "position_model_revision": resolved_position_revision,
            "pooling": pooling,
            "input_scale_uv": self.input_scale_uv,
            "adapter_checkpoint": str(adapter_checkpoint) if adapter_checkpoint else None,
            "adapter_sha256": adapter_sha256,
            "adapter_method": None if adapter_metadata is None else adapter_metadata.get("method"),
        }
        self.cache_signature = (
            f"reve:{model_name}@{resolved_model_revision}:"
            f"positions={position_model_name}@{resolved_position_revision}:"
            f"pool={pooling}:scale={self.input_scale_uv:g}:v3"
        )
        if adapter_sha256 is not None:
            self.cache_signature += f":adapter={adapter_sha256}"

    @property
    def pretrained_query(self) -> NDArray[np.float32]:
        """Return REVE's released global query token for probe initialization."""
        return self.model.cls_query_token.detach().float().cpu().numpy().reshape(-1).astype(np.float32)

    def summarize_cached(self, features: NDArray[np.floating]) -> NDArray[np.float32]:
        """Pool cached tokens with the released query for tractable drift metrics."""
        tokens = np.asarray(features, dtype=np.float32)
        if tokens.ndim != 3:
            return tokens.astype(np.float32, copy=False)
        query = self.pretrained_query
        scores = np.einsum("ntd,d->nt", tokens, query) / np.sqrt(tokens.shape[-1])
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True)
        return np.einsum("nt,ntd->nd", weights, tokens).astype(np.float32)

    def transform(
        self,
        x_uv: NDArray[np.floating],
        channel_names: Sequence[str],
        sfreq: float,
    ) -> NDArray[np.float32]:
        import torch

        if not np.isclose(sfreq, 200.0):
            raise ValueError(f"REVE requires 200 Hz input, received {sfreq}")
        x = np.asarray(x_uv, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected (trials, channels, time), got {x.shape}")

        with torch.inference_mode():
            positions = self.position_bank(list(channel_names)).float().to(self.device)
        if positions.shape[0] != len(channel_names):
            available = set(self.position_bank.get_all_positions())
            missing = [name for name in channel_names if name not in available]
            raise ValueError(f"REVE position bank does not contain channels: {missing}")

        outputs: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], self.batch_size):
                batch_np = x[start : start + self.batch_size] / self.input_scale_uv
                batch = torch.from_numpy(batch_np).to(self.device)
                batch_pos = positions.unsqueeze(0).expand(batch.shape[0], -1, -1)
                tokens = self.model(batch, batch_pos)
                if self.pooling == "attention":
                    embedding = self.model.attention_pooling(tokens)
                elif self.pooling == "mean":
                    embedding = tokens.mean(dim=(1, 2))
                else:
                    embedding = tokens.flatten(1, 2)
                array = embedding.float().cpu().numpy()
                if self.pooling == "tokens":
                    array = array.astype(np.float16)
                outputs.append(array)
        dtype = np.float16 if self.pooling == "tokens" else np.float32
        return np.concatenate(outputs, axis=0).astype(dtype, copy=False)


def build_encoder(experiment_config: dict) -> Encoder:
    encoder_name = str(experiment_config.get("encoder", "bandpower")).casefold()
    if encoder_name == "bandpower":
        return BandpowerEncoder()
    if encoder_name == "reve":
        return FrozenREVEEncoder(
            model_name=experiment_config.get("model_name", "brain-bzh/reve-base"),
            position_model_name=experiment_config.get("position_model_name", "brain-bzh/reve-positions"),
            device=experiment_config.get("device", "auto"),
            batch_size=int(experiment_config.get("batch_size", 32)),
            pooling=experiment_config.get("reve_pooling", "attention"),
            input_scale_uv=float(experiment_config.get("reve_input_scale_uv", 100.0)),
            model_revision=experiment_config.get("model_revision"),
            position_model_revision=experiment_config.get("position_model_revision"),
            adapter_checkpoint=experiment_config.get("adapter_checkpoint"),
        )
    raise ValueError(f"Unknown encoder: {encoder_name}")
