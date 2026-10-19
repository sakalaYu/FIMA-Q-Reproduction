"""Run a matched W3A3 test of task-weighted attention relations.

Worker 1 is the fixed-rotation control, worker 2 uses uniform relation KL,
and worker 3 weights query/head relations with gradients from FIMA-Q's existing
Fisher refreshes. The launcher needs only Python's standard library.
"""

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess

import run_headwise_experiments as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "relation_v1"
VARIANTS = ("control", "uniform", "fisher")
DEFAULTS = dict(relation_weight=0.1, relation_queries=16,
                relation_start=0.2, relation_full=0.6)


def run_name(variant, seed):
    return f"headwise_relation_w3a3_seed{seed}_{variant}_seeded_full_{PROTOCOL}"


def command(variant, args):
    result = base.command(3, args.seed, True, args.dataset, args.num_workers)
    result.extend(["--attention-relation", "none" if variant == "control" else variant])
    if variant != "control":
        for key in DEFAULTS:
            result.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        result.extend(["--relation-protocol", PROTOCOL])
    return result


def runs_for_seed(args):
    candidates = list((ROOT / "logs").glob("*headwise_relation*.log"))
    candidates += list((ROOT / "checkpoints/quant_result").glob(
        "*headwise_relation*/output.log"))
    found = {}
    for path in candidates:
        run = base.read_run(path)
        if run is None or run["seed"] != args.seed or run["bits"] != 3:
            continue
        if run["rotation_mode"] != "all" or not run["python_seeded"]:
            continue
        mode = run["args"].get("attention_relation", "none")
        variant = "control" if mode == "none" else mode
        if variant not in VARIANTS:
            continue
        if [item[0] for item in run["calibration_hashes"]] != ["128", "1024"]:
            continue
        if run["args"].get("relation_protocol", PROTOCOL) != PROTOCOL:
            continue
        if variant != "control" and any(
                run["args"].get(key) != getattr(args, key) for key in DEFAULTS):
            continue
        run["variant"] = variant
        report = ROOT / "checkpoints/quant_result" / run["run_id"] / "attention_relation.json"
        if variant == "control":
            run["relation_blocks"] = []
        elif report.is_file():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                blocks = data.get("blocks", [])
                if len(blocks) != 12 or {row.get("block") for row in blocks} != {
                        f"blocks.{i}" for i in range(12)}:
                    continue
                if data.get("options", {}).get("protocol") != PROTOCOL:
                    continue
                run["relation_blocks"] = blocks
            except (OSError, ValueError, TypeError):
                continue
        else:
            continue
        found[run["run_id"]] = run
    return sorted(found.values(), key=lambda row: row["run_id"])


def latest(runs, variant, args):
    matches = [row for row in runs if row["variant"] == variant
               and row["seed"] == args.seed
               and row["args"].get("num_workers") == args.num_workers
               and base.dataset_path(row["args"]["dataset"]) == base.dataset_path(args.dataset)
               and (variant == "control" or all(
                   row["args"].get(key) == getattr(args, key) for key in DEFAULTS))]
    return matches[-1] if matches else None


def summarize(args, write=True):
    runs = runs_for_seed(args)
    control = latest(runs, "control", args)
    with redirect_stdout(io.StringIO()):
        references = base.available_runs()
    fixed = base.latest_run(references, 3, args.seed, True, args.dataset)
    rows = []
    for variant in VARIANTS:
        run = latest(runs, variant, args)
        if run is None:
            print(f"{variant}: incomplete")
            continue
        matched_control = control is not None and all(
            run[key] == control[key] for key in ("cfg", "calibration_hashes"))
        matched_fixed = fixed is not None and all(
            run[key] == fixed[key] for key in ("cfg", "calibration_hashes"))
        matched_control = matched_control and run["args"].get("num_workers") == control["args"].get("num_workers")
        matched_fixed = matched_fixed and run["args"].get("num_workers") == fixed["args"].get("num_workers")
        row = dict(variant=variant, run=run,
                   delta_control_pp=(run["top1"] - control["top1"]
                                     if matched_control and variant != "control" else 0.0
                                     if variant == "control" else None),
                   delta_fixed_rotation_pp=(run["top1"] - fixed["top1"]
                                            if matched_fixed else None),
                   matched_control=bool(matched_control), matched_fixed=bool(matched_fixed))
        if variant != "control":
            row["active_blocks"] = sum(bool(block.get("active_steps")) for block in run["relation_blocks"])
            row["mean_weight_min"] = sum(block["weight_min"] for block in run["relation_blocks"]) / 12
            row["mean_weight_max"] = sum(block["weight_max"] for block in run["relation_blocks"]) / 12
        rows.append(row)
        delta_control = (f"{row['delta_control_pp']:+.3f} pp"
                         if row["delta_control_pp"] is not None else "N/A (control/config mismatch)")
        delta_fixed = (f"{row['delta_fixed_rotation_pp']:+.3f} pp"
                       if row["delta_fixed_rotation_pp"] is not None else "N/A (reference/config mismatch)")
        print(f"{variant}: Top-1 {run['top1']:.3f}; vs control {delta_control}; "
              f"vs prior fixed rotation {delta_fixed}")
        if variant != "control":
            print(f"  attention relation active in {row['active_blocks']}/12 blocks; "
                  f"mean query/head weight range {row['mean_weight_min']:.3f}..{row['mean_weight_max']:.3f}")
    report = dict(protocol=PROTOCOL, seed=args.seed, control=control,
                  prior_fixed_rotation=fixed, rows=rows, count=len(rows), expected_count=3)
    if write:
        folder = ROOT / "checkpoints/attention_relation"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"summary_seed{args.seed}_{PROTOCOL}.json"
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(target)
        print(f"Summary: {target}")
    return report


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS + ("summary",), default=None)
    parser.add_argument("--worker", type=int, default=1, help="1=control, 2=uniform, 3=fisher")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset", default="../imagenet_fimaq")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-slots", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.num_workers < 0 or args.seed < 0:
        parser.error("seed and num-workers must be nonnegative")
    if args.relation_weight <= 0 or args.relation_queries < 1:
        parser.error("relation weight and query count must be positive")
    if not 0 <= args.relation_start < args.relation_full <= 1:
        parser.error("Require 0 <= relation-start < relation-full <= 1")
    if args.variant == "summary":
        summarize(args, write=not args.dry_run)
        return
    if args.variant is None:
        if not 1 <= args.worker <= args.workers <= 3:
            parser.error("Require 1 <= worker <= workers <= 3")
        variant = VARIANTS[args.worker - 1] if args.workers == 3 else VARIANTS[args.worker - 1]
    else:
        variant = args.variant
    cmd = command(variant, args)
    name = run_name(variant, args.seed)
    if args.dry_run:
        print(f"{name}; GPU {args.gpu}, slots={args.gpu_slots}")
        print(shlex.join(cmd))
        return
    if os.name != "posix":
        parser.error("Actual runs require Linux; use --dry-run on Windows")
    dataset = base.dataset_path(args.dataset)
    if not all((dataset / split).is_dir() for split in ("train", "val")):
        parser.error(f"Missing train/val directories under {dataset}")
    checks = (("test_quant.py", "--attention-relation"),
              ("test_quant.py", "Python random seeded:"),
              ("utils/datasets.py", "FIMAQ_CAPTURE_CALIBRATION_HASH"),
              ("utils/orthogonal_vit.py", "def apply_headwise_hadamard"),
              ("utils/block_recon.py", "attention_relation_options"),
              ("utils/attention_relation.py", PROTOCOL))
    for relative, marker in checks:
        source = ROOT / relative
        if not source.is_file() or marker not in source.read_text(encoding="utf-8"):
            parser.error(f"Sync {relative} before launching")
    with base.job_lock(name):
        existing = latest(runs_for_seed(args), variant, args)
        if args.resume and existing:
            print(f"Skipping completed run: {existing['path']}")
            summarize(args)
            return
        with base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
            before = {row["run_id"] for row in runs_for_seed(args)}
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid, FIMAQ_RUN_NAME=name,
                       FIMAQ_CAPTURE_CALIBRATION_HASH="1")
            folder = ROOT / "checkpoints/attention_relation"
            folder.mkdir(parents=True, exist_ok=True)
            console = folder / f"{name}_console.log"
            print(shlex.join(cmd), flush=True)
            with console.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1,
                                           encoding="utf-8", errors="replace")
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        handle.write(line)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
        new = [row for row in runs_for_seed(args) if row["run_id"] not in before]
        if latest(new, variant, args) is None:
            raise RuntimeError("No completed run with all 12 attention block reports was found")
    summarize(args)


if __name__ == "__main__":
    main()
