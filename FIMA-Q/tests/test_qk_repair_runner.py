"""Keep repaired runs separate and ensure contrasts isolate intended factors."""
from pathlib import Path
import sys
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_qk_repair_experiments as runner


class RepairRunnerTests(unittest.TestCase):
    def test_three_windows_partition_each_suite_without_overlap(self):
        for suite, total in (("seed3407", 6), ("extra-seeds", 12), ("all", 18)):
            jobs = [job for worker in (1, 2, 3) for job in runner.planned_jobs(suite, worker, 3)]
            self.assertEqual(len(jobs), total)
            self.assertEqual(len(set(jobs)), total)

    def test_both_losses_are_crossed_with_rotation_capacity(self):
        args = runner.get_parser().parse_args([])
        for arm in ("B_mse", "B_dplr", "C_mse", "C_dplr"):
            cmd = runner.command(3407, arm, args)
            def value(flag): return cmd[cmd.index(flag)+1]
            self.assertEqual(value("--qk-joint-reconstruction-loss"), runner.ARMS[arm][1])
            self.assertEqual(value("--qk-joint-protocol"), "v2_integer_zp")
            self.assertEqual(value("--qk-joint-rotation-rank"), "0")
            self.assertEqual(value("--optim-metric"), "fisher_dplr")
            self.assertEqual(value("--recon-iters"), "20000")

    def test_mismatched_calibration_inputs_are_excluded_from_summary(self):
        common = dict(cfg={"k": 5}, args={"num_workers": 8, "dataset": "data"})
        records = {
            "seed3407_control.json": dict(status="completed", fingerprint="f", run=dict(common, top1=64., calibration_hashes=["a"])),
            "seed3407_C_dplr.json": dict(status="completed", fingerprint="f", run=dict(common, top1=65., calibration_hashes=["b"]))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, value in records.items(): runner.atomic_json(path/name, value)
            result = runner.summarize(path, "f", write=False)
            self.assertEqual(result["statistics"]["C_dplr_minus_control"]["count"], 0)
            self.assertIsNone(result["contrasts"][0]["delta_pp"])

    def test_a_changed_code_fingerprint_is_not_pooled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner.atomic_json(path/"seed3407_control.json", dict(status="completed", fingerprint="old", run={}))
            self.assertEqual(runner.summarize(path, "new", write=False)["contrasts"], [])


if __name__ == "__main__": unittest.main()
