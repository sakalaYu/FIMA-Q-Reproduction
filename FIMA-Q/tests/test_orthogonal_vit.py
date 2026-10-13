"""Numerical checks for QK/VO folding. Requires PyTorch, but no timm or GPU."""

import copy
import importlib.util
from pathlib import Path
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    spec = importlib.util.spec_from_file_location(
        "orthogonal_vit", Path(__file__).resolve().parents[1] / "utils/orthogonal_vit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 2
            self.qkv = nn.Linear(8, 24)
            self.proj = nn.Linear(8, 8)

        def forward(self, x):
            batch, tokens, _ = x.shape
            q, k, v = self.qkv(x).reshape(batch, tokens, 3, 2, 4).permute(2, 0, 3, 1, 4)
            weights = ((q @ k.transpose(-1, -2)) / 2).softmax(dim=-1)
            return self.proj((weights @ v).transpose(1, 2).reshape(batch, tokens, 8))

    class ToyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Module(), nn.Module()])
            for block in self.blocks:
                block.attn = Attention()

        def forward(self, x):
            # Accept both ordinary tokens and the deterministic verification image.
            x = x.flatten(1)[:, :32].reshape(x.shape[0], 4, 8)
            for block in self.blocks:
                x = x + block.attn(x)
            return x.mean(dim=1)


@unittest.skipIf(torch is None, "PyTorch unavailable; run this test in the server fimaq environment")
class RotationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = ToyViT().eval()
        self.inputs = torch.randn(3, 4, 8)

    def test_every_mode_preserves_outputs_and_parameter_count(self):
        expected = self.model(self.inputs)
        count = sum(p.numel() for p in self.model.parameters())
        for mode in ("all", "qk", "vo"):
            with self.subTest(mode=mode):
                model = copy.deepcopy(self.model)
                info = module.apply_headwise_hadamard(model, verify=True, mode=mode)
                self.assertTrue(torch.allclose(model(self.inputs), expected, atol=2e-6, rtol=2e-6))
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                self.assertEqual(info["mode"], mode)

    def test_partial_modes_only_change_the_requested_weights_and_biases(self):
        for mode in ("qk", "vo"):
            model = copy.deepcopy(self.model)
            module.apply_headwise_hadamard(model, mode=mode)
            for before_block, after_block in zip(self.model.blocks, model.blocks):
                before, after = before_block.attn, after_block.attn
                for branch in range(3):
                    rows = slice(branch * 8, (branch + 1) * 8)
                    should_change = branch < 2 if mode == "qk" else branch == 2
                    self.assertEqual(not torch.equal(before.qkv.weight[rows], after.qkv.weight[rows]), should_change)
                    self.assertEqual(not torch.equal(before.qkv.bias[rows], after.qkv.bias[rows]), should_change)
                self.assertEqual(torch.equal(before.proj.weight, after.proj.weight), mode == "qk")
                self.assertTrue(torch.equal(before.proj.bias, after.proj.bias))

    def test_default_matches_all_and_composing_qk_vo(self):
        default, explicit, composed = [copy.deepcopy(self.model) for _ in range(3)]
        module.apply_headwise_hadamard(default)
        module.apply_headwise_hadamard(explicit, mode="all")
        module.apply_headwise_hadamard(composed, mode="qk")
        module.apply_headwise_hadamard(composed, mode="vo")
        for key, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]))
            self.assertTrue(torch.equal(value, composed.state_dict()[key]))

    def test_verification_preserves_rng_and_training_state(self):
        self.model.train()
        state = torch.random.get_rng_state().clone()
        module.apply_headwise_hadamard(self.model, mode="qk")
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(self.model.training)

    def test_unknown_mode_rejected_before_mutation(self):
        state = copy.deepcopy(self.model.state_dict())
        with self.assertRaises(ValueError):
            module.apply_headwise_hadamard(self.model, mode="invalid")
        for key, value in state.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
