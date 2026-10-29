"""Verify causal contrasts, loss identity and three-window task coverage."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_qk_controlled_experiments as controlled


class ControlledTests(unittest.TestCase):
    def setUp(self):
        self.args = controlled.get_parser().parse_args([])

    def fake_runs(self):
        scores = dict(B_mse=64.0, B_dplr=64.2, C_mse=64.1, C_dplr=64.5,
                      A_dplr=63.9, C_lowdim=64.7, C_lowdim_gate=64.8)
        runs = []
        for index, job in enumerate(controlled.JOBS):
            args = controlled.job_args(job, self.args)
            flags = dict(controlled.qk.DEFAULTS)
            flags.update(qk_joint_variant=job.variant, qk_joint_protocol=job.protocol,
                         qk_joint_rotation_rank=job.rank, qk_joint_min_relative_kl=job.gate,
                         qk_joint_reconstruction_loss=job.loss, num_workers=8,
                         dataset=self.args.dataset)
            runs.append(dict(seed=args.seed, bits=3, cfg={"k": 5},
                             calibration_hashes=[["128", "a", "b"], ["1024", "c", "d"]],
                             args=flags, run_id=f"{index:03}_{job.label}", top1=scores[job.label],
                             qk_blocks=[dict(accepted=True) for _ in range(12)]))
        return runs

    def test_three_windows_cover_exactly_seven_unique_settings(self):
        groups = [controlled.selected_jobs("all", worker, 3) for worker in (1, 2, 3)]
        labels = [job.label for group in groups for job in group]
        self.assertEqual(len(labels), 7)
        self.assertEqual(set(labels), set(controlled.SUITES["all"]))
        self.assertEqual(len({controlled.qk.name(job.variant, 3407, job.protocol,
                                               job.rank, job.gate, job.loss)
                              for job in controlled.JOBS}), 7)

    def test_causal_suite_and_final_reconstruction_are_controlled(self):
        jobs = controlled.selected_jobs("causal")
        self.assertEqual({(job.variant, job.loss) for job in jobs},
                         {("B", "mse"), ("B", "dplr"), ("C", "mse"), ("C", "dplr")})
        for job in controlled.JOBS:
            command = controlled.qk.command(job.variant, controlled.job_args(job, self.args))
            self.assertEqual(command[command.index("--optim-metric") + 1], "fisher_dplr")
            self.assertEqual(command[command.index("--recon-iters") + 1], "20000")
            self.assertEqual(command[command.index("--qk-joint-reconstruction-loss") + 1], job.loss)

    def test_factorial_and_gate_contrasts_use_matched_runs(self):
        with patch.object(controlled.qk, "available_runs", return_value=self.fake_runs()), \
                patch.object(controlled.qk.base, "available_runs", return_value=[]), \
                redirect_stdout(io.StringIO()):
            report = controlled.summarize(self.args, write=False)
        results = {row["label"]: row["delta_top1_pp"] for row in report["contrasts"]}
        self.assertAlmostEqual(results["rotation_at_mse"], 0.1)
        self.assertAlmostEqual(results["rotation_at_dplr"], 0.3)
        self.assertAlmostEqual(results["five_percent_gate"], 0.1)
        self.assertAlmostEqual(report["rotation_loss_interaction_pp"], 0.2)

    def test_input_mismatch_excludes_a_causal_contrast(self):
        runs = self.fake_runs()
        next(run for run in runs if run["run_id"].endswith("C_dplr"))["calibration_hashes"] = []
        with patch.object(controlled.qk, "available_runs", return_value=runs), \
                patch.object(controlled.qk.base, "available_runs", return_value=[]), \
                redirect_stdout(io.StringIO()):
            report = controlled.summarize(self.args, write=False)
        row = next(row for row in report["contrasts"] if row["label"] == "rotation_at_dplr")
        self.assertIsNone(row["delta_top1_pp"])
        self.assertIsNone(report["rotation_loss_interaction_pp"])


if __name__ == "__main__":
    unittest.main()
