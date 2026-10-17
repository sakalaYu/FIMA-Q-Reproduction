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

    def add(self, filename, variant=None, blocks=12, top1=65.15, tensor_hash="b", rotation=True):
        extra = "num_workers=8, "
        if variant:
            extra += f"qk_joint_variant='{variant}', "
            extra += ", ".join(f"{key}={value!r}" for key, value in runner.DEFAULTS.items()) + ", "
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

    def test_input_mismatch_does_not_report_a_gain(self):
        self.add("001_headwise_rotation.log", top1=64.650)
        self.add("002_qkjoint_A.log", variant="A", tensor_hash="c")
        with redirect_stdout(io.StringIO()):
            report = runner.summarize(self.args, write=False)
        self.assertIsNone(report["rows"][0]["delta_fixed_pp"])

    def test_joint_variant_cannot_pollute_the_fixed_rotation_summary(self):
        self.add("001_headwise_qkjoint_A.log", variant="A")
        self.assertEqual(runner.base.available_runs(), [])

    def test_commands_use_the_full_recipe_and_variants_do_not_alias(self):
        for variant in ("A", "B", "C"):
            cmd = runner.command(variant, self.args)
            self.assertEqual(cmd[cmd.index("--qk-joint-variant") + 1], variant)
            for flag, value in (("--k", "5"), ("--recon-iters", "20000"),
                                ("--optim-size", "1024"), ("--seed", "3407")):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertEqual(len({runner.name(variant, 3407) for variant in ("A", "B", "C")}), 3)


if __name__ == "__main__":
    unittest.main()
