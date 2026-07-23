import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from gaugeeeg.cli import build_parser
from gaugeeeg.config import load_config, with_overrides
from gaugeeeg.mojepa import (
    OPERATOR_BASE_DIM,
    gauge_reference_groups,
    measurement_operator_code,
)

CHANNELS = (
    "FC3",
    "FC4",
    "C3",
    "C4",
    "CP3",
    "CP4",
    "Cz",
    "CPz",
    "FC1",
    "FC2",
    "C1",
    "C2",
    "CP1",
    "CP2",
    "Fz",
    "Pz",
    "F3",
    "F4",
    "F1",
    "F2",
    "FC5",
    "FC6",
    "C5",
    "C6",
    "CP5",
    "CP6",
    "P3",
    "P4",
    "PO3",
    "PO4",
    "O1",
    "O2",
    "Fp1",
    "Fp2",
    "F7",
    "F8",
    "FT7",
    "FT8",
    "T7",
    "T8",
    "TP7",
    "TP8",
    "P7",
    "P8",
    "PO7",
    "PO8",
    "Oz",
    "Iz",
    "AF3",
    "AF4",
    "AF7",
    "AF8",
    "F5",
    "F6",
    "FCz",
    "Cz2",
    "CPz2",
    "P1x",
    "P2x",
    "POz",
    "O9",
    "O10",
    "T9",
    "T10",
)


class MeasurementOperatorTests(unittest.TestCase):
    def test_code_distinguishes_gauge_from_lossy_montage(self):
        car = measurement_operator_code("car", CHANNELS)
        pz = measurement_operator_code("pz", CHANNELS)
        native = measurement_operator_code("native16@pz", CHANNELS)
        self.assertEqual(car.shape, (OPERATOR_BASE_DIM + len(CHANNELS),))
        self.assertEqual(float(car[7]), 0.0)
        self.assertEqual(float(pz[8]), 1.0)
        self.assertEqual(float(native[7]), 1.0)
        self.assertAlmostEqual(float(native[6]), 16.0 / len(CHANNELS))
        self.assertEqual(int(native[OPERATOR_BASE_DIM:].sum()), 16)
        self.assertFalse(np.array_equal(car, pz))

    def test_gauge_groups_never_mix_montages(self):
        views = ("car", "cz", "native32@car", "native32@pz", "native16@car")
        self.assertEqual(gauge_reference_groups(views), ((0, 1), (2, 3)))

    def test_cli_and_config_accept_adapter_workflow(self):
        adapt = build_parser().parse_args(
            ["adapt-mojepa", "--config", "configs/reve_gauge_mojepa_poc.yaml", "--seed", "7"]
        )
        self.assertEqual(adapt.command, "adapt-mojepa")
        run = build_parser().parse_args(
            [
                "run",
                "--config",
                "configs/reve_gauge_mojepa_poc.yaml",
                "--adapter-checkpoint",
                "/tmp/adapter.pt",
            ]
        )
        self.assertEqual(run.adapter_checkpoint, "/tmp/adapter.pt")
        changed = with_overrides(
            load_config("configs/reve_gauge_mojepa_poc.yaml"),
            adapter_checkpoint="/tmp/adapter.pt",
        )
        self.assertEqual(changed["experiment"]["adapter_checkpoint"], "/tmp/adapter.pt")


class LoRATests(unittest.TestCase):
    def test_end_to_end_synthetic_adaptation_writes_reloadable_adapter(self):
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest("torch is not installed")

        from gaugeeeg.datasets import EEGDataset
        from gaugeeeg.mojepa import run_mojepa_adaptation

        class FakePositionBank(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(_commit_hash="fake-position-sha")

            def forward(self, names):
                offset = torch.arange(len(names), dtype=torch.float32).unsqueeze(1)
                return torch.cat([offset, offset * 0.0, offset * 0.0], dim=1)

        class ReveAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(8)
                self.to_qkv = nn.Linear(8, 24, bias=False)
                self.to_out = nn.Linear(8, 8, bias=False)

            def forward(self, value):
                query, key, content = self.to_qkv(self.norm(value)).chunk(3, dim=-1)
                return self.to_out((query + key + content) / 3.0)

        class FakeReve(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch = nn.Linear(1, 8)
                self.transformer = nn.Module()
                self.transformer.layers = nn.ModuleList(
                    [nn.ModuleList([ReveAttention(), nn.Linear(8, 8)]) for _ in range(3)]
                )
                self.config = types.SimpleNamespace(_commit_hash="fake-model-sha")

            def forward(self, signal, positions):
                del positions
                value = self.patch(signal.mean(dim=-1, keepdim=True))
                for attention, feed_forward in self.transformer.layers:
                    value = value + attention(value)
                    value = value + torch.tanh(feed_forward(value))
                return value

            def attention_pooling(self, value):
                return value.mean(dim=1)

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                del kwargs
                return FakePositionBank() if "positions" in name else FakeReve()

        rng = np.random.default_rng(12)
        labels = np.tile(np.arange(4, dtype=np.int64), 4)
        dataset = EEGDataset(
            x_uv=rng.normal(size=(16, len(CHANNELS), 16)).astype(np.float32),
            y=labels,
            subjects=np.asarray([1] * 8 + [61] * 8, dtype=np.int64),
            channel_names=CHANNELS,
            sfreq=200.0,
            label_names=("left", "right", "both", "feet"),
        )
        config = copy.deepcopy(load_config("configs/reve_gauge_mojepa_poc.yaml"))
        config["adaptation"].update(
            {
                "views": ["car", "pz", "native16@car", "native16@pz"],
                "lora_last_n_blocks": 2,
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "batch_size": 4,
                "epochs": 1,
                "patience": 1,
                "max_train_trials": 8,
                "max_val_trials": 8,
            }
        )
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModel = FakeAutoModel
        with tempfile.TemporaryDirectory() as temporary:
            previous_transformers = sys.modules.get("transformers")
            sys.modules["transformers"] = fake_transformers
            try:
                with mock.patch("gaugeeeg.mojepa.load_physionet_mi", return_value=dataset):
                    summary = run_mojepa_adaptation(
                        config,
                        objective="gauge_mojepa",
                        device="cpu",
                        seed=3,
                        output_dir=temporary,
                    )
            finally:
                if previous_transformers is None:
                    sys.modules.pop("transformers", None)
                else:
                    sys.modules["transformers"] = previous_transformers
            checkpoint = Path(temporary) / "adapter.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(summary["method"], "gauge_mojepa")
            self.assertEqual(summary["adapter_parameters"], 192)
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertTrue(summary["training_head_discarded"])
            self.assertEqual(len(payload["adapter_state"]), 8)
            self.assertTrue(
                any(
                    value.abs().sum().item() > 0.0
                    for name, value in payload["adapter_state"].items()
                    if name.endswith("lora_b")
                )
            )

    def test_reve_style_split_projection_names_are_supported(self):
        try:
            from torch import nn
        except ImportError:
            self.skipTest("torch is not installed")

        from gaugeeeg.mojepa import inject_reve_lora

        class ReveAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_qkv = nn.Linear(8, 24, bias=False)
                self.to_out = nn.Linear(8, 8, bias=False)

        class ReveStyleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.layers = nn.ModuleList(
                    [nn.ModuleList([ReveAttention(), nn.Linear(8, 8)]) for _ in range(3)]
                )

        model = ReveStyleEncoder()
        targets = inject_reve_lora(model, rank=2, alpha=2.0, last_n_blocks=2)
        self.assertEqual(
            targets,
            [
                "transformer.layers.1.0.to_qkv.weight",
                "transformer.layers.1.0.to_out.weight",
                "transformer.layers.2.0.to_qkv.weight",
                "transformer.layers.2.0.to_out.weight",
            ],
        )

    def test_fused_attention_lora_has_gradients_and_round_trips(self):
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest("torch is not installed")

        from gaugeeeg.mojepa import (
            inject_reve_lora,
            load_mojepa_adapter,
            lora_parameter_count,
            lora_state_dict,
            set_lora_enabled,
        )

        class ToyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.MultiheadAttention(8, 2, batch_first=True) for _ in range(3)])

            def forward(self, value):
                for layer in self.layers:
                    value, _ = layer(value, value, value, need_weights=False)
                return value

        torch.manual_seed(2)
        model = ToyEncoder()
        frozen_state = copy.deepcopy(model.state_dict())
        targets = inject_reve_lora(model, rank=2, alpha=4.0, last_n_blocks=2)
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.layers[0].parameters()))
        self.assertGreater(lora_parameter_count(model), 0)
        value = torch.randn(4, 5, 8)
        loss = model(value).square().mean()
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(parameter.grad is not None for parameter in trainable))
        torch.optim.SGD(trainable, lr=0.2).step()
        enabled_output = model(value).detach()
        set_lora_enabled(model, False)
        base_output = model(value).detach()
        self.assertFalse(torch.allclose(enabled_output, base_output))
        set_lora_enabled(model, True)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "adapter.pt"
            torch.save(
                {
                    "metadata": {
                        "rank": 2,
                        "alpha": 4.0,
                        "last_n_blocks": 2,
                        "targets": targets,
                        "method": "gauge_mojepa",
                    },
                    "adapter_state": lora_state_dict(model),
                },
                checkpoint,
            )
            restored = ToyEncoder()
            restored.load_state_dict(frozen_state)
            metadata = load_mojepa_adapter(restored, checkpoint)
            self.assertEqual(metadata["method"], "gauge_mojepa")
            self.assertEqual(set(lora_state_dict(model)), set(lora_state_dict(restored)))
            self.assertTrue(torch.allclose(model(value), restored(value)))


if __name__ == "__main__":
    unittest.main()
