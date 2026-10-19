"""Run W3A3 Q/K variants A/B/C in three tmux windows (standard library only)."""

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
DEFAULTS = dict(qk_joint_steps=300, qk_joint_batch_size=8, qk_joint_holdout=128,
                qk_joint_lr=0.003, qk_joint_eval_every=50)
PROTOCOL = "v2_integer_zp"


def command(variant, args):
    result = base.command(3, args.seed, True, args.dataset, args.num_workers)
    result.extend(["--qk-joint-variant", variant, "--qk-joint-protocol", PROTOCOL])
    for key in DEFAULTS:
        result.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    return result


def name(variant, seed):
    return f"qkjoint_w3a3_seed{seed}_{variant}_seeded_full_{PROTOCOL}"


def available_runs():
    paths = list((ROOT / "logs").glob("*qkjoint*.log"))
    paths += list((ROOT / "checkpoints/quant_result").glob("*qkjoint*/output.log"))
    result = {}
    for path in paths:
        run = base.read_run(path)
        if run is None or run["args"].get("qk_joint_variant") not in ("A", "B", "C"):
            continue
        if [item[0] for item in run["calibration_hashes"]] != ["128", "1024"] or not run["python_seeded"]:
            continue
        if run["rotation_mode"] != "all" or run["bits"] != 3:
            continue
        blocks = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("QK joint result: "):
                blocks.append(json.loads(line.split(": ", 1)[1]))
        if {block["block"] for block in blocks} != {f"blocks.{i}" for i in range(12)} or len(blocks) != 12:
            continue
        run["qk_blocks"] = blocks
        result[run["run_id"]] = run
    return sorted(result.values(), key=lambda run: run["run_id"])


def latest(runs, variant, args, protocol=None):
    matches = [run for run in runs if run["seed"] == args.seed
               and run["args"].get("qk_joint_variant") == variant
               and run["args"].get("num_workers") == args.num_workers
               and base.dataset_path(run["args"]["dataset"]) == base.dataset_path(args.dataset)
               and all(run["args"].get(key) == getattr(args, key) for key in DEFAULTS)
               and (protocol is None or run["args"].get("qk_joint_protocol") == protocol)]
    return matches[-1] if matches else None


def summarize(args, write=True):
    # Reuse historical parsing and require actual inputs/configs to match.
    with redirect_stdout(io.StringIO()):
        historical = base.available_runs()
    baseline = base.latest_run(historical, 3, args.seed, False, args.dataset)
    fixed = base.latest_run(historical, 3, args.seed, True, args.dataset)
    reference = fixed if fixed is not None else baseline
    rows = []
    runs = available_runs()
    for variant in ("A", "B", "C"):
        run = latest(runs, variant, args)
        if run is None:
            print(f"{variant}: incomplete")
            continue
        valid = reference is not None and all(
            run[key] == reference[key] for key in ("cfg", "calibration_hashes"))
        valid = valid and run["args"].get("num_workers") == reference["args"].get("num_workers")
        row = dict(variant=variant, protocol=run["args"].get("qk_joint_protocol", "v1"),
                   run=run, matched_reference=bool(valid),
                   accepted_blocks=sum(block["accepted"] for block in run["qk_blocks"]),
                   calibration_seconds=sum(block["elapsed_seconds"] for block in run["qk_blocks"]),
                   delta_fixed_pp=run["top1"] - fixed["top1"] if valid and fixed else None,
                   delta_baseline_pp=run["top1"] - baseline["top1"] if valid and baseline
                   and run["cfg"] == baseline["cfg"]
                   and run["calibration_hashes"] == baseline["calibration_hashes"] else None)
        rows.append(row)
        delta = f"{row['delta_fixed_pp']:+.3f} pp" if row["delta_fixed_pp"] is not None else "N/A (reference mismatch/missing)"
        print(f"{variant}: Top-1 {run['top1']:.3f}; versus fixed all: {delta}; "
              f"accepted blocks {row['accepted_blocks']}/12; protocol {row['protocol']}")
    report = dict(seed=args.seed, fixed_rotation=fixed, baseline=baseline, rows=rows,
                  count=len(rows), expected_count=3)
    if write:
        directory = ROOT / "checkpoints/qk_joint"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"summary_seed{args.seed}.json"
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(target)
        print(f"Summary: {target}")
    return report


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A", "B", "C", "summary"), required=True)
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
    if args.num_workers < 0 or any(getattr(args, key) <= 0 for key in DEFAULTS):
        parser.error("QK sizes, steps, interval and learning rate must be positive; num-workers >= 0")
    if args.qk_joint_holdout >= base.RECIPE["optim_size"]:
        parser.error("QK holdout must be smaller than optim_size=1024")
    if args.variant == "summary":
        summarize(args, write=not args.dry_run)
        return
    cmd = command(args.variant, args)
    run_name = name(args.variant, args.seed)
    if args.dry_run:
        print(f"{run_name}; GPU {args.gpu}, slots={args.gpu_slots}")
        print(shlex.join(cmd))
        return
    if os.name != "posix":
        parser.error("Actual runs require Linux; use --dry-run on Windows")
    dataset = base.dataset_path(args.dataset)
    if not all((dataset / split).is_dir() for split in ("train", "val")):
        parser.error(f"Missing train/val directories under {dataset}")
    for path, marker in (("test_quant.py", "--qk-joint-protocol"),
                         ("utils/block_recon.py", "calibrate_qk_block"),
                         ("utils/datasets.py", "FIMAQ_CAPTURE_CALIBRATION_HASH"),
                         ("utils/qk_joint.py", PROTOCOL)):
        source = ROOT / path
        if not source.is_file() or marker not in source.read_text(encoding="utf-8"):
            parser.error(f"Sync {path} before launching")
    with base.job_lock(run_name):
        existing = latest(available_runs(), args.variant, args, protocol=PROTOCOL)
        if args.resume and existing:
            print(f"Skipping completed run: {existing['path']}")
            summarize(args)
            return
        with base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
            before = {run["run_id"] for run in available_runs()}
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid, FIMAQ_RUN_NAME=run_name,
                       FIMAQ_CAPTURE_CALIBRATION_HASH="1")
            directory = ROOT / "checkpoints/qk_joint"
            directory.mkdir(parents=True, exist_ok=True)
            # Preserve stderr/tracebacks as a small artifact even on failure.
            console = directory / f"{run_name}_console.log"
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
        new = [run for run in available_runs() if run["run_id"] not in before]
        if latest(new, args.variant, args, protocol=PROTOCOL) is None:
            raise RuntimeError("No newly completed run with 12 QK block reports and calibration hashes")
    summarize(args)


if __name__ == "__main__":
    main()
