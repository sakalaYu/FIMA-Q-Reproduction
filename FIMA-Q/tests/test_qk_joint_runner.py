"""Result accounting checks for the three independent QK experiments."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qk_joint_experiments as runner
from test_headwise_experiments import log_text


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "logs").mkdir()
        self.patches = [patch.object(runner, "ROOT", self.root), patch.object(runner.base, "ROOT", self.root)]
        for item in self.patches:
            item.start()
        self.args = runner.get_parser().parse_args(["--variant", "A"])

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def add(self, filename, variant=None, blocks=12, top1=65.15, tensor_hash="b", rotation=True,
            loss=None, protocol=None):
        extra = "num_workers=8, "
        if variant:
            extra += f"qk_joint_variant='{variant}', "
            extra += ", ".join(f"{key}={value!r}" for key, value in runner.DEFAULTS.items()) + ", "
            if loss is not None:
                extra += f"qk_joint_reconstruction_loss={loss!r}, "
            if protocol is not None:
                extra += f"qk_joint_protocol={protocol!r}, "
        text = log_text(bits=3, rotation=rotation, top1=top1, tensor_hash=tensor_hash)
        text = text.replace("Namespace(", "Namespace(" + extra, 1)
        if variant:
            for index in range(blocks):
                text += "\nQK joint result: " + json.dumps(dict(
                    block=f"blocks.{index}", variant=variant, accepted=True, elapsed_seconds=1))
        path = self.root / "logs" / filename
        path.write_text(text, encoding="utf-8")
        return path

    def test_incomplete_calibration_is_not_resumed_as_completed(self):
        self.add("001_qkjoint.log", variant="A", blocks=11)
        self.assertEqual(runner.available_runs(), [])

    def test_result_compares_against_matched_fixed_rotation(self):
        self.add("001_headwise_baseline.log", top1=63.358, rotation=False)
        self.add("002_headwise_rotation.log", top1=64.650)
        self.add("003_qkjoint_A.log", variant="A", top1=65.150)
        with redirect_stdout(io.StringIO()):
            report = runner.summarize(self.args, write=False)
        self.assertEqual(report["count"], 1)
        self.assertAlmostEqual(report["rows"][0]["delta_fixed_pp"], 0.5)
        self.assertAlmostEqual(report["rows"][0]["delta_baseline_pp"], 1.792)

    def test_old_method_uses_the_prior_fixed_run_when_a_later_control_exists(self):
        self.add("001_headwise_rotation.log", top1=64.650)
        self.add("002_qkjoint_A.log", variant="A", top1=64.388)
        self.add("003_headwise_rotation.log", top1=64.524)
        with redirect_stdout(io.StringIO()):
            report = runner.summarize(self.args, write=False)
        self.assertAlmostEqual(report["rows"][0]["delta_fixed_pp"], -0.262)
        self.assertAlmostEqual(report["rows"][0]["fixed_reference"]["top1"], 64.650)

    def test_input_mismatch_does_not_report_a_gain(self):
        self.add("001_headwise_rotation.log", top1=64.650)
        self.add("002_qkjoint_A.log", variant="A", tensor_hash="c")
        with redirect_stdout(io.StringIO()):
            report = runner.summarize(self.args, write=False)
        self.assertIsNone(report["rows"][0]["delta_fixed_pp"])

    def test_joint_variant_cannot_pollute_the_fixed_rotation_summary(self):
        self.add("001_headwise_qkjoint_A.log", variant="A")
        self.assertEqual(runner.base.available_runs(), [])

    def test_corrected_protocol_does_not_resume_an_old_completed_run(self):
        self.add("001_qkjoint_C.log", variant="C")
        runs = runner.available_runs()
        self.assertIsNotNone(runner.latest(runs, "C", self.args))
        self.assertIsNone(runner.latest(runs, "C", self.args, protocol=runner.PROTOCOL))

    def test_commands_use_the_full_recipe_and_variants_do_not_alias(self):
        for variant in ("A", "B", "C"):
            cmd = runner.command(variant, self.args)
            self.assertEqual(cmd[cmd.index("--qk-joint-variant") + 1], variant)
            for flag, value in (("--k", "5"), ("--recon-iters", "20000"),
                                ("--optim-size", "1024"), ("--seed", "3407")):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertEqual(len({runner.name(variant, 3407) for variant in ("A", "B", "C")}), 3)

    def test_low_rank_run_has_distinct_identity_and_parameters(self):
        self.args.protocol = "v3_lowrank_gate"
        self.args.rotation_rank = 4
        self.args.min_relative_kl = 0.05
        cmd = runner.command("C", self.args)
        self.assertEqual(cmd[cmd.index("--qk-joint-protocol") + 1], self.args.protocol)
        self.assertEqual(cmd[cmd.index("--qk-joint-rotation-rank") + 1], "4")
        self.assertEqual(cmd[cmd.index("--qk-joint-min-relative-kl") + 1], "0.05")
        self.assertNotEqual(runner.name("C", 3407),
                            runner.name("C", 3407, self.args.protocol, 4, 0.05))

    def test_loss_controls_have_distinct_identity_and_resume_matching(self):
        self.add("001_qkjoint_B.log", variant="B", protocol=runner.PROTOCOL, loss="mse")
        self.args.reconstruction_loss = "dplr"
        self.assertIsNone(runner.latest(runner.available_runs(), "B", self.args, runner.PROTOCOL))
        self.args.reconstruction_loss = "mse"
        self.assertIsNotNone(runner.latest(runner.available_runs(), "B", self.args, runner.PROTOCOL))
        self.assertNotEqual(runner.name("C", 3407, reconstruction_loss="mse"),
                            runner.name("C", 3407, reconstruction_loss="dplr"))

    def test_explicit_dplr_reuses_the_previous_auto_c_protocol(self):
        self.add("001_qkjoint_C.log", variant="C", protocol=runner.PROTOCOL)
        self.args.reconstruction_loss = "dplr"
        self.assertIsNotNone(runner.latest(runner.available_runs(), "C", self.args, runner.PROTOCOL))


if __name__ == "__main__":
    unittest.main()
