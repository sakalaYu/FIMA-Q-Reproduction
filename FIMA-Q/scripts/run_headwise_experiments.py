"""Run and summarize full rotation pairs, sharded across up to four tmux windows.

Only standard-library dependencies are required by this launcher. Actual model
runs use the existing Linux CUDA environment. --gpu-slots explicitly controls
how many of these jobs may share one physical GPU.
"""

import argparse
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
METRIC = re.compile(
    r"^\s*\* Prec@1 ([\d.]+) Prec@5 ([\d.]+) Loss ([\d.]+) Time ([\d.]+)",
    re.MULTILINE,
)
RECIPE = dict(calib_size=128, calib_batch_size=32, optim_size=1024,
              optim_batch_size=32, recon_iters=20000, k=5)
FIXED_CONFIG = dict(optim_metric="fisher_dplr", optim_mode="qdrop", drop_prob=0.5,
                    p1=1.0, p2=1.0, dis_mode="q", temp=20, calib_metric="mse",
                    qconv_a_bit=8, matmul_head_channel_wise=True,
                    token_channel_wise=True, eq_n=128, search_round=3, keep_gpu=True)


def planned_jobs(suite, worker=1, workers=1, include_3407=False):
    if not 1 <= worker <= workers <= 4:
        raise ValueError("Require 1 <= worker <= workers <= 4")
    jobs = []
    if suite in ("all", "w4-seeds"):
        seeds = (3408, 3409, 3407) if include_3407 else (3408, 3409)
        jobs.extend((4, seed, rotation) for seed in seeds for rotation in (False, True))
    if suite in ("all", "w3-pair"):
        jobs.extend((3, 3407, rotation) for rotation in (False, True))
    if suite == "w4-3407":
        jobs.extend((4, 3407, rotation) for rotation in (False, True))
    return jobs[worker - 1::workers]


def run_name(bits, seed, rotation):
    return f"headwise_w{bits}a{bits}_seed{seed}_{'rotation' if rotation else 'baseline'}_seeded_full"


def read_run(path):
    """Read a completed full experiment without executing its Namespace text."""
    text = path.read_text(encoding="utf-8", errors="replace")
    namespace = next((s for s in text.splitlines() if s.startswith("Namespace(")), None)
    if namespace is None or "- finished the process." not in text:
        return None
    try:
        call = ast.parse(namespace, mode="eval").body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "Namespace":
            return None
        args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        cfg = {}
        for key, value in re.findall(r"^([a-z][a-z_0-9]*): (.+)$", text, re.MULTILINE):
            try:
                cfg[key] = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                cfg[key] = value.strip()
        metrics = METRIC.findall(text)
    except (ValueError, SyntaxError, AttributeError):
        return None
    if not (args.get("calibrate") and args.get("optimize")) or len(metrics) != 3:
        return None
    if args.get("model") != "vit_small" or args.get("val_batch_size") != 64:
        return None
    if any(cfg.get(key) != value for key, value in RECIPE.items()):
        return None
    if cfg.get("optim_metric") != "fisher_dplr" or cfg.get("optim_mode") != "qdrop":
        return None
    bits = cfg.get("w_bit")
    if bits not in (3, 4) or cfg.get("a_bit") != bits:
        return None
    if any(cfg.get(key) != value for key, value in FIXED_CONFIG.items()):
        return None
    if cfg.get("qhead_a_bit") != bits:
        return None
    seeded = f"Python random seeded: {args.get('seed')}" in text
    hashes = re.findall(r"Calibration cache: count=(\d+) sample_sha256=([a-f0-9]{64}) "
                        r"tensor_sha256=([a-f0-9]{64})", text)
    elapsed = re.search(r"Block reconstruction elapsed seconds: ([\d.]+)", text)
    return dict(path=str(path), args=args, cfg=cfg, bits=bits,
                run_id=path.parent.name if path.name == "output.log" else path.stem,
                python_seeded=seeded, calibration_hashes=hashes,
                seed=args["seed"], rotation=bool(args.get("headwise_hadamard", False)),
                top1=float(metrics[-1][0]), top5=float(metrics[-1][1]),
                loss=float(metrics[-1][2]), calibration_top1=float(metrics[0][0]),
                reconstruction_seconds=float(elapsed.group(1)) if elapsed else None)


def available_runs():
    # Also find completed output.log files if the final copy into logs was interrupted.
    paths = list((ROOT / "logs").glob("*headwise*.log"))
    paths += list((ROOT / "checkpoints/quant_result").glob("*headwise*/output.log"))
    runs = [run for path in paths if (run := read_run(path)) is not None]
    # logs/ and output.log can be two copies of the same run.
    unique = {run["run_id"]: run for run in runs}
    return sorted(unique.values(), key=lambda run: run["run_id"])


def dataset_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def latest_run(runs, bits, seed, rotation, dataset, seeded=True):
    matches = [r for r in runs if r["bits"] == bits and r["seed"] == seed
               and r["rotation"] == rotation
               and r["python_seeded"] == seeded
               and dataset_path(r["args"]["dataset"]) == dataset_path(dataset)]
    if seeded:
        matches = [r for r in matches if [h[0] for h in r["calibration_hashes"]] == ["128", "1024"]]
    return matches[-1] if matches else None


def summary(dataset, write=True):
    runs = available_runs()
    report = {}
    print("\nPaired results (percentage points; latest completed run per setting):")
    for bits, seeds in ((4, (3407, 3408, 3409)), (3, (3407,))):
        pairs = []
        for seed in seeds:
            baseline = latest_run(runs, bits, seed, False, dataset)
            rotation = latest_run(runs, bits, seed, True, dataset)
            if baseline is None or rotation is None:
                print(f"W{bits}A{bits} seed {seed}: incomplete pair")
                continue
            if (baseline["cfg"] != rotation["cfg"] or
                    baseline["args"].get("num_workers") != rotation["args"].get("num_workers")):
                print(f"W{bits}A{bits} seed {seed}: config/worker mismatch; excluded")
                continue
            if baseline["calibration_hashes"] != rotation["calibration_hashes"]:
                print(f"W{bits}A{bits} seed {seed}: augmented calibration inputs differ; excluded")
                continue
            delta = rotation["top1"] - baseline["top1"]
            pairs.append(dict(seed=seed, baseline=baseline, rotation=rotation,
                              delta_top1_pp=delta))
            print(f"W{bits}A{bits} seed {seed}: {baseline['top1']:.3f} -> "
                  f"{rotation['top1']:.3f}; delta {delta:+.3f} pp")
        deltas = [p["delta_top1_pp"] for p in pairs]
        group = dict(pairs=pairs, count=len(pairs), expected_count=len(seeds),
                     mean_delta_pp=statistics.mean(deltas) if deltas else None,
                     sample_std_delta_pp=statistics.stdev(deltas) if len(deltas) > 1 else None,
                     mean_baseline_top1=statistics.mean(p["baseline"]["top1"] for p in pairs) if pairs else None,
                     mean_rotation_top1=statistics.mean(p["rotation"]["top1"] for p in pairs) if pairs else None)
        report[f"w{bits}a{bits}"] = group
        if deltas:
            std = f"{group['sample_std_delta_pp']:.3f}" if len(deltas) > 1 else "N/A (one pair)"
            print(f"  {len(deltas)}/{len(seeds)} pairs: mean delta "
                  f"{group['mean_delta_pp']:+.3f} pp; sample std {std} pp")
    legacy = [r for r in runs if not r["python_seeded"]]
    report["legacy_reference"] = legacy
    if legacy:
        print("Historical runs without Python random seeding are reference only; excluded from means.")
    if write:
        target = ROOT / "checkpoints/headwise_experiments/summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        # Each writer has its own temporary file, so two suites can finish together.
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(target)
        print(f"Summary: {target}")
    return report


def command(bits, seed, rotation, dataset, workers):
    result = [sys.executable, "-u", "test_quant.py", "--model", "vit_small",
              "--config", f"./configs/{bits}bit/best.py", "--dataset", str(dataset),
              # 'cuda' preserves the parent's CUDA_VISIBLE_DEVICES selection.
              # test_quant.py rewrites that variable for '--device cuda:N'.
              "--device", "cuda", "--calibrate", "--optimize",
              "--val-batch-size", "64", "--num-workers", str(workers),
              "--seed", str(seed), "--optim-metric", "fisher_dplr",
              "--optim-mode", "qdrop", "--drop-prob", "0.5",
              "--p1", "1", "--p2", "1", "--dis-mode", "q"]
    for key, value in RECIPE.items():
        result.extend(["--" + key.replace("_", "-"), str(value)])
    if rotation:
        result.append("--headwise-hadamard")
    return result


@contextmanager
def gpu_lock(gpu, slots=1):
    if os.name != "posix":
        raise RuntimeError("Actual runs require Linux. Use --dry-run on Windows.")
    import fcntl
    result = subprocess.run(
        ["nvidia-smi", "--id=" + gpu, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True)
    uuids = result.stdout.strip().splitlines()
    if len(uuids) != 1 or not re.fullmatch(r"GPU-[a-fA-F0-9-]+", uuids[0].strip()):
        raise RuntimeError("--gpu must identify one physical NVIDIA GPU index or UUID")
    uuid = uuids[0].strip()
    directory = ROOT / "checkpoints/headwise_experiments/locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{uuid}.lock").open("a") as handle:
        print(f"Waiting for GPU {gpu} ({uuid}); configured slots={slots}...", flush=True)
        # Shared runs cannot overlap an exclusive run launched by this script.
        fcntl.flock(handle, fcntl.LOCK_EX if slots == 1 else fcntl.LOCK_SH)
        slot_handle = None
        try:
            if slots > 1:
                while slot_handle is None:
                    for slot in range(slots):
                        candidate = (directory / f"{uuid}.slot{slot}.lock").open("a")
                        try:
                            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            slot_handle = candidate
                            break
                        except BlockingIOError:
                            candidate.close()
                    if slot_handle is None:
                        time.sleep(1)
            print(f"Acquired GPU {gpu} slot; starting one model process.", flush=True)
            yield uuid
        finally:
            if slot_handle is not None:
                fcntl.flock(slot_handle, fcntl.LOCK_UN)
                slot_handle.close()
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def job_lock(name):
    import fcntl
    directory = ROOT / "checkpoints/headwise_experiments/locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True,
                        choices=("all", "w4-seeds", "w3-pair", "w4-3407", "summary"))
    parser.add_argument("--worker", type=int, default=1, help="1-based tmux window number")
    parser.add_argument("--workers", type=int, default=1, help="Total windows, at most 4")
    parser.add_argument("--include-3407", action="store_true",
                        help="Also rerun W4A4 seed 3407 with fixed Python random seeding")
    parser.add_argument("--gpu", default="0", help="Physical GPU index from nvidia-smi, or GPU UUID")
    parser.add_argument("--gpu-slots", type=int, default=1, choices=range(1, 5),
                        help="Maximum concurrent jobs on this GPU; use the same value in every window")
    parser.add_argument("--dataset", default="../imagenet_fimaq")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Skip matching completed runs; failed runs restart")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without GPU/data access or writes")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.suite == "summary":
        summary(args.dataset, write=not args.dry_run)
        return
    try:
        jobs = planned_jobs(args.suite, args.worker, args.workers, args.include_3407)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        for bits, seed, rotation in jobs:
            print(f"{run_name(bits, seed, rotation)}; GPU {args.gpu}, slots={args.gpu_slots}")
            print(shlex.join(command(bits, seed, rotation, args.dataset, args.num_workers)))
        return
    if os.name != "posix":
        parser.error("Actual runs require Linux; use --dry-run on Windows")
    entry_source = (ROOT / "test_quant.py").read_text(encoding="utf-8")
    data_source = (ROOT / "utils/datasets.py").read_text(encoding="utf-8")
    if "Python random seeded:" not in entry_source or "FIMAQ_CAPTURE_CALIBRATION_HASH" not in data_source:
        parser.error("Sync test_quant.py and utils/datasets.py before starting this launcher")
    dataset = dataset_path(args.dataset)
    if not all((dataset / split).is_dir() for split in ("train", "val")):
        parser.error(f"Missing train/val directories under {dataset}")
    for bits, seed, rotation in jobs:
        name = run_name(bits, seed, rotation)
        with job_lock(name):
            existing = latest_run(available_runs(), bits, seed, rotation, dataset)
            if args.resume and existing:
                print(f"Skipping completed run: {existing['path']}", flush=True)
                continue
            with gpu_lock(args.gpu, args.gpu_slots) as uuid:
                before = {run["run_id"] for run in available_runs()}
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = uuid
                env["FIMAQ_RUN_NAME"] = name
                env["FIMAQ_CAPTURE_CALIBRATION_HASH"] = "1"
                # Preserve relative data paths for logs synced to another host.
                cmd = command(bits, seed, rotation, args.dataset, args.num_workers)
                print(f"\nStarting {name}\n{shlex.join(cmd)}", flush=True)
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            new_runs = [r for r in available_runs() if r["run_id"] not in before]
            completed = latest_run(new_runs, bits, seed, rotation, dataset)
            if completed is None:
                raise RuntimeError("No NEW completed, seeded full run with calibration hashes was found. "
                                   "Sync test_quant.py and utils/datasets.py too.")
            summary(dataset)
    summary(dataset)


if __name__ == "__main__":
    main()
