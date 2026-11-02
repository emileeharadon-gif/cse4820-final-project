"""CNN contract checks. Run: python -m unittest discover -s tests -v"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.cnn import CNN
from models.cnnmax import CNNMax
from models.cnn_common import BirdCNN, load_prepared


class CNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_architectures_have_matched_initial_weights_and_parameters(self):
        torch.manual_seed(42)
        plain = CNN()
        torch.manual_seed(42)
        pooled = CNNMax()
        self.assertEqual(sum(p.numel() for p in plain.parameters()), 355084)
        self.assertEqual(sum(p.numel() for p in pooled.parameters()), 355084)
        for a, b in zip(plain.parameters(), pooled.parameters()):
            torch.testing.assert_close(a, b)
        self.assertFalse(any(isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d,
                                               nn.AdaptiveAvgPool2d, nn.Linear))
                             for layer in plain.modules()))
        self.assertEqual(sum(isinstance(layer, nn.MaxPool2d) for layer in pooled.modules()), 3)

    def test_both_models_produce_class_scores_and_receive_gradients(self):
        for cls in [CNN, CNNMax]:
            with self.subTest(model=cls.__name__):
                torch.manual_seed(42)
                model = cls(num_classes=2, input_shape=(16, 16), dropout=0)
                logits = model(torch.randn(3, 1, 16, 16))
                self.assertEqual(tuple(logits.shape), (3, 2))
                nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 0])).backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_model_can_learn_a_tiny_training_batch(self):
        torch.manual_seed(42)
        model = CNN(num_classes=2, input_shape=(16, 16), dropout=0)
        x = torch.randn(4, 1, 16, 16)
        y = torch.tensor([0, 0, 1, 1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        initial = criterion(model(x), y).item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        self.assertLess(criterion(model(x), y).item(), initial * 0.1)

    def test_checkpoint_preserves_predictions(self):
        torch.manual_seed(42)
        model = CNNMax(num_classes=2, input_shape=(16, 16)).eval()
        architecture = {"num_classes": 2, "input_shape": [16, 16], "dropout": 0.3, "use_max_pool": True}
        inputs = torch.randn(2, 1, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"architecture": architecture, "state": model.state_dict()}, path)
            payload = torch.load(path, weights_only=True)
            restored = BirdCNN(**payload["architecture"]).eval()
            restored.load_state_dict(payload["state"])
            with torch.inference_mode():
                torch.testing.assert_close(model(inputs), restored(inputs))

    def write_fixture(self, folder, leak=False, mismatched_stats=False):
        config = {"label_map": {"species_a": 0, "species_b": 1}, "n_species": 2,
                  "logmel_shape": [16, 16], "mel_mean": 0.0, "mel_std": 1.0}
        (folder / "config.json").write_text(json.dumps(config))
        splits = np.repeat(["train", "val", "test"], 4)
        labels = np.tile([0, 0, 1, 1], 3)
        recordings = np.array([f"{split}-{label}" for split, label in zip(splits, labels)])
        if leak:
            recordings[4] = recordings[0]
        np.savez_compressed(folder / "features.npz", logmel=np.ones((12, 16, 16), dtype=np.float32),
                            y=labels, split=splits, recording_id=recordings,
                            mel_mean=0.0, mel_std=2.0 if mismatched_stats else 1.0)

    def test_loading_adds_channel_and_preserves_split_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            data = load_prepared(root)
            self.assertEqual(tuple(data.x.shape), (12, 1, 16, 16))
            self.assertEqual(data.summary()["test"], {"clips": 4, "recordings": 2})

    def test_loading_rejects_recording_leakage_and_wrong_normalization(self):
        for case in ["leak", "normalization"]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_fixture(root, leak=case == "leak", mismatched_stats=case == "normalization")
                with self.assertRaises(ValueError):
                    load_prepared(root)

if __name__ == "__main__":
    unittest.main()
