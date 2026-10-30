"""CPU numerical and interrupted-resume checks on real W3A3 quantizers."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from test_qk_joint import Block, linear
from utils.local_teacher import (activation_scales, quant_mode, quant_state,
    vit_prefix, vit_suffix, distillation, estimate_anchor, dplr_loss, clamp_scales)
import run_local_teacher as entry
from run_local_teacher_experiments import planned_jobs


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Identity()
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(8)
        self.head = linear(8, 6)
        self.default_cfg = {}

    def _pos_embed(self, x):
        return x

    def forward_head(self, x):
        return self.head(x.mean(1))

    def forward(self, x):
        return vit_suffix(self, self.blocks[0](vit_prefix(self, x, 0)), 0)


class LocalTeacherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)
        self.student = TinyViT().eval()
        self.teacher = copy.deepcopy(self.student)
        for module in self.teacher.modules():
            if hasattr(module, "mode"):
                module.mode = "raw"
        for model in (self.student, self.teacher):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        quant_mode(self.student)
        self.x = torch.randn(10, 5, 8)

    def test_quant_suffix_keeps_forward_and_input_gradient(self):
        original = self.student(self.x)
        with quant_state(self.student, ste=True):
            h = vit_prefix(self.student, self.x, 0)
            y = self.student.blocks[0](h).detach().requires_grad_(True)
            actual = vit_suffix(self.student, y, 0)
            self.assertTrue(torch.equal(actual, original))
            distillation(actual, self.teacher(self.x)).backward()
            self.assertGreater(float(y.grad.norm()), 0)
            self.assertTrue(all(p.grad is None for p in self.student.parameters()))
        self.assertTrue(torch.equal(self.student(self.x), original))

    def test_fp_and_quant_teacher_paths_are_distinct(self):
        with torch.no_grad():
            h = vit_prefix(self.student, self.x, 0)
            target = self.teacher.blocks[0](h)
            fp = vit_suffix(self.teacher, target, 0)
            quant = vit_suffix(self.student, target, 0)
        self.assertGreater(float((fp - quant).abs().max()), 1e-4)
        self.assertTrue(torch.equal(h, self.x))

    def test_standardized_kd_is_invariant_to_positive_affine_logit_changes(self):
        z = torch.randn(8, 6)
        t = torch.randn(8, 6)
        a = distillation(z, t, standardized=True)
        b = distillation(z * 3 + 17, t * 5 - 8, standardized=True)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_anchor_normalization_and_seed_are_fixed(self):
        with torch.no_grad():
            h = vit_prefix(self.student, self.x, 0)
            targets = self.teacher.blocks[0](h)
            logits = self.teacher(self.x)
        kwargs = dict(count=8, rank=2, batch_size=4, seed=6)
        a = estimate_anchor(self.student, self.teacher, 0, h, targets, logits, **kwargs)
        torch.randn(57)
        b = estimate_anchor(self.student, self.teacher, 0, h, targets, logits, **kwargs)
        self.assertTrue(torch.equal(a["gradient"], b["gradient"]))
        self.assertTrue(torch.isfinite(a["inverse"]).all())
        value = dplr_loss(self.student.blocks[0](h[:8]), targets[:8], a)
        self.assertAlmostEqual(float(value), 2.0, places=5)

    def test_scales_are_bounded_and_weight_zero_points_are_frozen(self):
        params = activation_scales(self.student.blocks[0])
        centers = {key: value.detach().clone() for key, value in params.items()}
        with torch.no_grad():
            for value in params.values():
                value.mul_(10)
        clamp_scales(params, centers, 0.1)
        self.assertTrue(all(torch.allclose(p, centers[key] * 1.1) for key, p in params.items()))
        self.assertFalse(any("w_quantizer" in key or "zero_point" in key for key in params))

    def test_three_workers_cover_screen_and_all_once(self):
        self.assertEqual([planned_jobs("screen", i, 3) for i in (1, 2, 3)],
                         [["E3"], ["E4"], ["E5"]])
        jobs = sum([planned_jobs("all", i, 3) for i in (1, 2, 3)], [])
        self.assertEqual(sorted(jobs), ["E1", "E2", "E3", "E4", "E5", "E6"])

    def test_resume_restores_optimizer_sampler_and_anchor_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "origin.pth"
            torch.save(self.student.state_dict(), checkpoint)
            torch.save({}, root / "baseline_predictions.pt")
            cfg = dict(checkpoint=str(checkpoint), blocks=[0, 1], train_count=8,
                steps_per_block=3, batch_size=4, eval_batch_size=4, relative_lr=0.001,
                scale_radius=0.1, fisher_count=8, fisher_rank=2, fisher_temperature=20.,
                seed=21, kd_temperature=2., local_weight=1., global_weight=1., save_every=1)
            with torch.no_grad():
                data = dict(images=self.x, global_logits=self.teacher(self.x))
            timm = types.ModuleType("timm")
            timm_data = types.ModuleType("timm.data")
            timm_data.resolve_data_config = lambda *args, **kwargs: {}
            timm_data.create_transform = lambda **kwargs: None
            fake_modules = {"timm": timm, "timm.data": timm_data}
            full = root / "full"
            resumed = root / "resumed"
            full.mkdir()
            resumed.mkdir()
            def fake_validation(*args, **kwargs):
                return dict(top1=50., top5=90., repaired=1, damaged=1, delta_pp=0.), {}
            class SimulatedInterruption(Exception):
                pass
            interrupted = [False]
            def interrupt(directory, event):
                if event["event"] == "train" and not interrupted[0]:
                    interrupted[0] = True
                    raise SimulatedInterruption()
            with patch.dict(sys.modules, fake_modules), patch.object(entry, "validation", fake_validation):
                with patch.object(entry, "record", lambda *args: None):
                    entry.train_arm(full, root, "E6", cfg, "test", copy.deepcopy(self.student),
                                    self.teacher, data, torch.device("cpu"), entry.time.perf_counter())
                with patch.object(entry, "record", interrupt):
                    with self.assertRaises(SimulatedInterruption):
                        entry.train_arm(resumed, root, "E6", cfg, "test", copy.deepcopy(self.student),
                                        self.teacher, data, torch.device("cpu"), entry.time.perf_counter())
                with patch.object(entry, "record", lambda *args: None):
                    entry.train_arm(resumed, root, "E6", cfg, "test", copy.deepcopy(self.student),
                                    self.teacher, data, torch.device("cpu"), entry.time.perf_counter())
            a = torch.load(full / "model.pth", map_location="cpu")
            b = torch.load(resumed / "model.pth", map_location="cpu")
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))


class ModelSourceTests(unittest.TestCase):
    def config(self, root):
        path = root / "student.pth"
        torch.save({"weight": torch.ones(2)}, path)
        return dict(checkpoint=str(path), teacher_checkpoint=None,
                    teacher_model="vit_small_patch16_224", output_root=str(root / "results"))

    def test_missing_student_and_explicit_missing_teacher_fail_without_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self.config(root)
            with patch.object(entry, "ROOT", root):
                cfg["teacher_checkpoint"] = str(root / "missing.safetensors")
                with self.assertRaisesRegex(FileNotFoundError, "Explicit FP"):
                    entry.resolve_inputs(cfg)
                cfg["checkpoint"] = str(root / "missing_student.pth")
                with self.assertRaisesRegex(FileNotFoundError, "student checkpoint missing"):
                    entry.resolve_inputs(cfg)

    def test_timm_teacher_is_materialized_once_without_vit_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self.config(root)
            calls = []
            fake_timm = types.ModuleType("timm")
            fake_timm.__version__ = "test"
            def create_model(name, pretrained):
                calls.append((name, pretrained))
                return nn.Linear(2, 3)
            fake_timm.create_model = create_model
            with patch.object(entry, "ROOT", root), patch.dict(sys.modules, {"timm": fake_timm}):
                a = entry.resolve_inputs(cfg)
                b = entry.resolve_inputs(cfg, allow_download=False)
                self.assertEqual(a, b)
                self.assertEqual(len(calls), 1)
                self.assertFalse((root / "checkpoints/vit_raw").exists())
                self.assertEqual(entry.teacher_state(a["teacher_checkpoint"])["weight"].shape, (3, 2))
                json.dumps(a)

    def test_safetensors_is_supported_and_quantized_teacher_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self.config(root)
            bad = root / "quantized.pth"
            torch.save({"a_quantizer.scale": torch.ones(1)}, bad)
            with self.assertRaisesRegex(ValueError, "cannot serve"):
                entry.teacher_state(bad)
            try:
                from safetensors.torch import save_file
            except ImportError:
                self.skipTest("safetensors not installed")
            path = root / "checkpoints/model.safetensors"
            path.parent.mkdir()
            save_file({"head.weight": torch.ones(3, 2)}, str(path))
            with patch.object(entry, "ROOT", root):
                resolved = entry.resolve_inputs(cfg, allow_download=False)
            self.assertEqual(resolved["teacher_checkpoint"], str(path.resolve()))
            self.assertTrue(torch.equal(entry.teacher_state(path)["head.weight"], torch.ones(3, 2)))

    def test_student_override_requires_corresponding_baseline(self):
        args = types.SimpleNamespace(checkpoint="B_mse.pth", teacher_checkpoint=None,
                                     expected_baseline_top1=None)
        with self.assertRaisesRegex(ValueError, "requires --expected-baseline-top1"):
            entry.apply_source_arguments({}, args)
        args.expected_baseline_top1 = 64.726
        cfg = entry.apply_source_arguments({}, args)
        self.assertEqual(cfg["checkpoint"], "B_mse.pth")
        self.assertEqual(cfg["expected_baseline_top1"], 64.726)


class EvaluationPolicyTests(unittest.TestCase):
    def test_legacy_validation_is_separate_from_cache_and_restores_backends(self):
        cfg = dict(eval_batch_size=32, validation_batch_size=64, validation_profile="legacy")
        original = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with entry.evaluation_backend(cfg) as policy:
                self.assertEqual(policy["batch_size"], 64)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertFalse(torch.backends.cudnn.deterministic)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                raise RuntimeError("interrupted")
        actual = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                  torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(actual, original)

    def test_accuracy_is_integer_counted_and_pairing_is_preserved(self):
        from torch.utils.data import TensorDataset
        logits = torch.eye(6)[:5] * 5
        labels = torch.tensor([0, 1, 0, 3, 4])
        data = TensorDataset(logits, labels)
        cfg = dict(dataset="unused", eval_batch_size=32, validation_batch_size=2,
                   validation_profile="strict", num_workers=0)
        baseline = dict(labels=labels, correct=torch.tensor([False, True, True, True, True]))
        with patch("torchvision.datasets.ImageFolder", return_value=data):
            result, predictions = entry.validation(nn.Identity(), cfg, None, torch.device("cpu"), baseline)
        self.assertEqual(result["top1"], 80.0)
        self.assertEqual(result["repaired"], 1)
        self.assertEqual(result["damaged"], 1)
        self.assertEqual(result["delta_pp"], 0.0)
        self.assertTrue(torch.equal(predictions["labels"], labels))

    def test_audit_has_fixed_profiles_and_does_not_choose_highest_score(self):
        fake_timm = types.ModuleType("timm")
        fake_data = types.ModuleType("timm.data")
        fake_data.resolve_data_config = lambda *a, **k: {}
        fake_data.create_transform = lambda **k: None
        teacher = types.SimpleNamespace(default_cfg={})
        calls = []
        def validate(student, cfg, transform, device, baseline):
            policy = entry.evaluation_policy(cfg)
            calls.append((policy["profile"], policy["batch_size"]))
            prediction = dict(predictions=torch.tensor([0, 1]))
            return dict(top1=50.0 + len(calls), evaluation_policy=policy), prediction
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(sys.modules, {"timm": fake_timm, "timm.data": fake_data}), \
                 patch.object(entry, "validation", validate), \
                 patch.object(entry, "evaluation_metadata", return_value={}), \
                 patch.object(entry, "record", lambda *a: None):
                cfg = dict(validation_profile="legacy", validation_batch_size=64)
                entry.audit_evaluation(Path(temporary), cfg, "test", None, teacher, torch.device("cpu"))
                self.assertEqual(cfg["validation_profile"], "legacy")
                self.assertTrue((Path(temporary) / "evaluation_audit.json").is_file())
        self.assertEqual(calls, [("legacy", 64), ("strict", 64), ("strict", 32)])
        self.assertEqual(planned_jobs("audit", 1, 1), ["audit"])


if __name__ == "__main__":
    unittest.main()
