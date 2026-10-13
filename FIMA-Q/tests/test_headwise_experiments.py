"""CPU-only checks for experiment result accounting; no model/GPU imports."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_headwise_experiments.py"
SPEC = importlib.util.spec_from_file_location("headwise_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def log_text(seed=3407, rotation=False, top1=76.2, complete=True, iterations=20000,
             drop_prob=0.5, seeded=True, tensor_hash="b", bits=4, mode=None):
    cfg = dict(runner.RECIPE, **runner.FIXED_CONFIG)
    cfg.update(w_bit=bits, a_bit=bits, qhead_a_bit=bits, drop_prob=drop_prob)
    cfg["recon_iters"] = iterations
    mode_arg = f", headwise_mode='{mode}'" if mode is not None else ""
    lines = [f"Namespace(model='vit_small', dataset='../imagenet_fimaq', "
             f"calibrate=True, optimize=True, val_batch_size=64, seed={seed}, "
             f"headwise_hadamard={rotation}{mode_arg})"]
    lines.extend(f"{key}: {value}" for key, value in cfg.items())
    if seeded:
        lines.append(f"Python random seeded: {seed}")
        for count in (128, 1024):
            lines.append(f"Calibration cache: count={count} sample_sha256={'a' * 64} "
                         f"tensor_sha256={tensor_hash * 64}")
    lines += [" * Prec@1 60.000 Prec@5 83.000 Loss 1.700 Time 60.000",
              " * Prec@1 85.000 Prec@5 96.000 Loss 0.600 Time 2.000",
              f" * Prec@1 {top1:.3f} Prec@5 94.000 Loss 0.880 Time 60.000"]
    if complete:
        lines.append("2026-10-10 19:00:00 - finished the process.")
    return "\n".join(lines)


class ResultAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "logs").mkdir()
        self.root_patch = patch.object(runner, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def add(self, filename, **kwargs):
        path = self.root / "logs" / filename
        path.write_text(log_text(**kwargs), encoding="utf-8")
        return path

    def test_uses_final_validation_not_training_accuracy(self):
        run = runner.read_run(self.add("headwise.log", top1=76.894))
        self.assertEqual(run["top1"], 76.894)
        self.assertEqual(run["calibration_top1"], 60.0)

    def test_excludes_unfinished_and_smoke_runs(self):
        self.assertIsNone(runner.read_run(self.add("headwise_partial.log", complete=False)))
        self.assertIsNone(runner.read_run(self.add("headwise_smoke.log", iterations=2)))

    def test_duplicate_run_does_not_count_as_an_independent_seed(self):
        self.add("001_headwise_old_rotation.log", rotation=True, top1=90.0)
        self.add("002_headwise_baseline.log", top1=76.0)
        self.add("003_headwise_rotation.log", rotation=True, top1=76.2)
        self.add("004_headwise_baseline.log", seed=3408, top1=75.0)
        self.add("005_headwise_rotation.log", seed=3408, rotation=True, top1=75.4)
        with redirect_stdout(io.StringIO()):
            report = runner.summary("../imagenet_fimaq", write=False)["w4a4"]
        self.assertEqual(report["count"], 2)
        self.assertAlmostEqual(report["mean_delta_pp"], 0.3)
        self.assertAlmostEqual(report["sample_std_delta_pp"], 0.1414213562)

    def test_mismatched_pair_is_excluded(self):
        self.add("001_headwise_baseline.log")
        self.add("002_headwise_rotation.log", rotation=True, drop_prob=1.0)
        with redirect_stdout(io.StringIO()):
            report = runner.summary("../imagenet_fimaq", write=False)
        self.assertEqual(report["w4a4"]["count"], 0)

    def test_different_augmented_inputs_are_excluded(self):
        self.add("001_headwise_baseline.log")
        self.add("002_headwise_rotation.log", rotation=True, tensor_hash="c")
        with redirect_stdout(io.StringIO()):
            report = runner.summary("../imagenet_fimaq", write=False)
        self.assertEqual(report["w4a4"]["count"], 0)

    def test_historical_seed_is_not_mixed_with_corrected_seeds(self):
        self.add("001_headwise_baseline.log", seeded=False)
        self.add("002_headwise_rotation.log", rotation=True, seeded=False)
        self.add("003_headwise_baseline.log", seed=3408, top1=76.0)
        self.add("004_headwise_rotation.log", seed=3408, rotation=True, top1=76.4)
        with redirect_stdout(io.StringIO()):
            report = runner.summary("../imagenet_fimaq", write=False)
        self.assertEqual(report["w4a4"]["count"], 1)
        self.assertEqual(report["w4a4"]["pairs"][0]["seed"], 3408)
        self.assertAlmostEqual(report["w4a4"]["mean_delta_pp"], 0.4)
        self.assertEqual(len(report["legacy_reference"]), 2)

    def test_partial_rotations_do_not_replace_all_rotation_or_resume_it(self):
        self.add("001_headwise_base.log", bits=3, top1=63.0)
        self.add("002_headwise_all.log", bits=3, rotation=True, top1=64.0)
        self.add("003_headwise_qk.log", bits=3, rotation=True, mode="qk", top1=63.5)
        self.add("004_headwise_vo.log", bits=3, rotation=True, mode="vo", top1=63.8)
        runs = runner.available_runs()
        self.assertEqual(runner.latest_run(runs, 3, 3407, True, "../imagenet_fimaq")["top1"], 64.0)
        self.assertEqual(runner.latest_run(runs, 3, 3407, "qk", "../imagenet_fimaq")["top1"], 63.5)
        with redirect_stdout(io.StringIO()):
            report = runner.summary("../imagenet_fimaq", write=False)
        self.assertEqual(report["w3a3"]["mean_delta_pp"], 1.0)
        self.assertEqual(report["w3a3_seed3407_ablation"]["count"], 4)
        rows = report["w3a3_seed3407_ablation"]["rows"]
        self.assertEqual([r["mode"] for r in rows], ["none", "qk", "vo", "all"])
        self.assertAlmostEqual(rows[1]["delta_top1_pp"], 0.5)

    def test_ablation_with_different_inputs_is_excluded(self):
        self.add("001_headwise_base.log", bits=3)
        self.add("002_headwise_qk.log", bits=3, rotation=True, mode="qk", tensor_hash="c")
        with redirect_stdout(io.StringIO()):
            report = runner.summary("../imagenet_fimaq", write=False)
        self.assertEqual(report["w3a3_seed3407_ablation"]["count"], 1)

    def test_three_w3_seeds_are_reported_as_three_pairs(self):
        for seed, delta in ((3407, 1.0), (3408, 0.5), (3409, 1.5)):
            self.add(f"{seed}_headwise_base.log", bits=3, seed=seed, top1=63.0)
            self.add(f"{seed}_headwise_all.log", bits=3, seed=seed, rotation=True, top1=63.0 + delta)
        with redirect_stdout(io.StringIO()):
            report = runner.summary("../imagenet_fimaq", write=False)["w3a3"]
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["expected_count"], 3)
        self.assertEqual(report["mean_delta_pp"], 1.0)
        self.assertEqual(report["sample_std_delta_pp"], 0.5)

    def test_w3_pair_is_separate_and_has_no_one_pair_std(self):
        self.add("001_headwise_baseline.log", bits=3, top1=64.0)
        self.add("002_headwise_rotation.log", bits=3, rotation=True, top1=65.0)
        with redirect_stdout(io.StringIO()):
            report = runner.summary("../imagenet_fimaq", write=False)
        self.assertEqual(report["w3a3"]["count"], 1)
        self.assertEqual(report["w3a3"]["mean_delta_pp"], 1.0)
        self.assertIsNone(report["w3a3"]["sample_std_delta_pp"])


class SchedulingTests(unittest.TestCase):
    def test_three_windows_keep_followup_pairs_together(self):
        parts = [runner.planned_jobs("w3-followup", i, 3) for i in range(1, 4)]
        self.assertEqual(parts, [
            [(3, 3408, False), (3, 3408, True)],
            [(3, 3409, False), (3, 3409, True)],
            [(3, 3407, "qk"), (3, 3407, "vo")],
        ])
        flattened = [job for part in parts for job in part]
        self.assertEqual(len(set(flattened)), 6)
        self.assertEqual(set(flattened), set(runner.planned_jobs("w3-followup")))

    def test_followup_covers_exactly_six_new_settings(self):
        expected = {(3, 3408, False), (3, 3408, True), (3, 3409, False),
                    (3, 3409, True), (3, 3407, "qk"), (3, 3407, "vo")}
        parts = [runner.planned_jobs("w3-followup", i, 4) for i in range(1, 5)]
        flattened = [job for part in parts for job in part]
        self.assertEqual(set(flattened), expected)
        self.assertEqual(len(flattened), 6)
        self.assertEqual(len(runner.planned_jobs("w3-seeds")), 4)
        self.assertEqual(len(runner.planned_jobs("w3-ablation")), 2)

    def test_partial_rotation_commands_and_names_are_distinct(self):
        names = [runner.run_name(3, 3407, mode) for mode in (False, True, "qk", "vo")]
        self.assertEqual(len(set(names)), 4)
        self.assertEqual(runner.run_name(3, 3407, "all"), names[1])
        for mode in ("qk", "vo"):
            cmd = runner.command(3, 3407, mode, "../imagenet_fimaq", 8)
            self.assertIn("--headwise-hadamard", cmd)
            self.assertEqual(cmd[cmd.index("--headwise-mode") + 1], mode)
        self.assertNotIn("--headwise-hadamard", runner.command(3, 3407, "none", "x", 8))

    def test_four_windows_cover_six_jobs_without_duplicates(self):
        parts = [runner.planned_jobs("all", worker=i, workers=4) for i in range(1, 5)]
        self.assertEqual([len(part) for part in parts], [2, 2, 1, 1])
        jobs = [job for part in parts for job in part]
        self.assertEqual(len(set(jobs)), 6)
        self.assertEqual(set(jobs), set(runner.planned_jobs("all")))
        self.assertEqual(parts[0], [(4, 3408, False), (3, 3407, False)])
        self.assertEqual(parts[1], [(4, 3408, True), (3, 3407, True)])

    def test_optional_corrected_3407_pair_is_explicit(self):
        self.assertEqual(len(runner.planned_jobs("all", include_3407=True)), 8)
        self.assertEqual(runner.planned_jobs("w4-3407"), [(4, 3407, False), (4, 3407, True)])

    def test_device_does_not_override_gpu_visibility(self):
        cmd = runner.command(3, 3407, True, "../imagenet_fimaq", 8)
        self.assertEqual(cmd[cmd.index("--device") + 1], "cuda")
        self.assertEqual(cmd[cmd.index("--config") + 1], "./configs/3bit/best.py")
        self.assertEqual(cmd[cmd.index("--k") + 1], "5")
        self.assertIn("--headwise-hadamard", cmd)

    def test_rejects_invalid_window_numbers(self):
        for worker, workers in ((0, 4), (5, 4), (1, 5)):
            with self.assertRaises(ValueError):
                runner.planned_jobs("all", worker, workers)


if __name__ == "__main__":
    unittest.main()
