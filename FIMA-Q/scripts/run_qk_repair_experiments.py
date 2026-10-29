"""Integer-zero-point Q/K audit; six arms, three tmux workers, stdlib only.

All arms keep fixed QKVO and the original 20k-step DPLR reconstruction.
Only the 300-step pre-calibration is varied. Historical runs are never resumed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

import run_headwise_experiments as base

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"control": ("none", "auto"), "B_mse": ("B", "mse"),
        "C_mse": ("C", "mse"), "A_dplr": ("A", "dplr"),
        "B_dplr": ("B", "dplr"), "C_dplr": ("C", "dplr")}
PROTOCOL = "qk_repair_v1"
PRECALIBRATION = dict(steps=300, batch_size=8, holdout=128, lr=0.003, eval_every=50)


def planned_jobs(suite, worker=1, workers=3):
    if not 1 <= worker <= workers <= 3:
        raise ValueError("Require 1 <= worker <= workers <= 3")
    seeds = {"seed3407": (3407,), "extra-seeds": (3408, 3409),
             "all": (3407, 3408, 3409), "summary": (3407, 3408, 3409)}[suite]
    jobs = [(seed, arm) for seed in seeds for arm in ARMS]
    return jobs[worker - 1::workers]


def command(seed, arm, args):
    variant, loss = ARMS[arm]
    cmd = base.command(3, seed, True, args.dataset, args.num_workers)
    cmd.extend(["--qk-joint-variant", variant, "--qk-joint-protocol", "v2_integer_zp",
                "--qk-joint-rotation-rank", "0", "--qk-joint-min-relative-kl", "0",
                "--qk-joint-reconstruction-loss", loss])
    for key, value in PRECALIBRATION.items():
        cmd.extend(["--qk-joint-" + key.replace("_", "-"), str(value)])
    return cmd


def source_fingerprint(args):
    paths = [ROOT/"test_quant.py", Path(__file__), ROOT/"scripts/run_headwise_experiments.py"]
    for name in ("utils", "configs", "quant_layers", "quantizers"):
        paths.extend((ROOT/name).rglob("*.py"))
    model = ROOT/"checkpoints/vit_raw/vit_small_patch16_224.bin"
    if not model.is_file():
        raise FileNotFoundError(f"Require the same local FP checkpoint for every arm: {model}")
    paths.append(model)
    files = {}
    for path in sorted(set(paths)):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.relative_to(ROOT).as_posix()] = digest.hexdigest()
    identity = dict(files=files, dataset=args.dataset, num_workers=args.num_workers,
                    precalibration=PRECALIBRATION, recipe=base.RECIPE, protocol=PROTOCOL)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), files


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def read_result(path):
    if not path.exists(): return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if value.get("status") == "completed" else None


def validate_run(path, seed, arm, args):
    run = base.read_run(path)
    variant, loss = ARMS[arm]
    if run is None or not run["python_seeded"] or run["seed"] != seed:
        raise RuntimeError(f"Missing a complete full-evaluation run in {path}")
    if run["bits"] != 3 or run["rotation_mode"] != "all" or [x[0] for x in run["calibration_hashes"]] != ["128", "1024"]:
        raise RuntimeError("Rotation or calibration hashes do not match the audit")
    if run["args"].get("qk_joint_variant") != variant or run["args"].get("qk_joint_reconstruction_loss") != loss:
        raise RuntimeError("Wrong pre-calibration variant/objective")
    if variant != "none":
        blocks = [json.loads(line.split(": ", 1)[1]) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.startswith("QK joint result: ")]
        if len(blocks) != 12 or {x["block"] for x in blocks} != {f"blocks.{i}" for i in range(12)}:
            raise RuntimeError("Expected exactly 12 completed pre-calibration blocks")
        if any(x.get("protocol") != "v2_integer_zp" or x.get("reconstruction_loss") != loss for x in blocks):
            raise RuntimeError("Wrong zero-point protocol or objective in block reports")
        run["accepted_blocks"] = sum(x["accepted"] for x in blocks)
        run["precalibration_seconds"] = sum(x["elapsed_seconds"] for x in blocks)
    run["path"] = path.relative_to(ROOT).as_posix()
    return run


def summarize(directory, fingerprint, write=True):
    rows = []
    contrasts = {"A_dplr_minus_control": ("A_dplr", "control"),
                 "scaling_with_DPLR": ("B_dplr", "A_dplr"),
                 "rotation_with_MSE": ("C_mse", "B_mse"),
                 "rotation_with_DPLR": ("C_dplr", "B_dplr"),
                 "DPLR_without_rotation": ("B_dplr", "B_mse"),
                 "DPLR_with_rotation": ("C_dplr", "C_mse"),
                 **{f"{arm}_minus_control": (arm, "control") for arm in ARMS if arm != "control"}}
    deltas = {name: [] for name in contrasts}
    for seed in (3407, 3408, 3409):
        records = {arm: read_result(directory/f"seed{seed}_{arm}.json") for arm in ARMS}
        records = {arm: r for arm, r in records.items() if r and r["fingerprint"] == fingerprint}
        print(f"Seed {seed}: " + "; ".join(f"{arm}={r['run']['top1']:.3f}" for arm, r in records.items()))
        for label, (left, right) in contrasts.items():
            if left not in records or right not in records: continue
            a, b = records[left]["run"], records[right]["run"]
            matched = (a["cfg"] == b["cfg"] and a["calibration_hashes"] == b["calibration_hashes"]
                       and a["args"]["num_workers"] == b["args"]["num_workers"]
                       and a["args"]["dataset"] == b["args"]["dataset"])
            delta = a["top1"] - b["top1"] if matched else None
            rows.append(dict(seed=seed, contrast=label, matched=matched, delta_pp=delta))
            if matched:
                deltas[label].append(delta)
                print(f"  {label}: {delta:+.3f} pp")
            else: print(f"  {label}: INPUT/CONFIG MISMATCH; excluded")
    interactions = []
    for seed in (3407, 3408, 3409):
        estimates = {r["contrast"]: r["delta_pp"] for r in rows if r["seed"] == seed and r["matched"]}
        if "rotation_with_MSE" in estimates and "rotation_with_DPLR" in estimates:
            four = [read_result(directory/f"seed{seed}_{arm}.json")["run"]
                    for arm in ("B_mse", "B_dplr", "C_mse", "C_dplr")]
            if all(run["cfg"] == four[0]["cfg"] and run["calibration_hashes"] == four[0]["calibration_hashes"]
                   and run["args"]["num_workers"] == four[0]["args"]["num_workers"]
                   and run["args"]["dataset"] == four[0]["args"]["dataset"] for run in four[1:]):
                interactions.append(dict(seed=seed, delta_pp=estimates["rotation_with_DPLR"]-estimates["rotation_with_MSE"]))
    summary = dict(fingerprint=fingerprint, contrasts=rows, rotation_loss_interactions=interactions, statistics={
        k: dict(count=len(v), mean_pp=statistics.mean(v) if v else None,
                sample_std_pp=statistics.stdev(v) if len(v) > 1 else None)
        for k, v in deltas.items()})
    if write: atomic_json(directory/"summary.json", summary)
    return summary


def run_job(seed, arm, args, directory, fingerprint):
    name = f"{PROTOCOL}_{fingerprint[:10]}_seed{seed}_{arm}"
    target = directory/f"seed{seed}_{arm}.json"
    with base.job_lock(name):
        if args.resume and (completed := read_result(target)):
            if completed["fingerprint"] == fingerprint:
                print(f"Skip completed {name}", flush=True)
                return
        with base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
            cmd = command(seed, arm, args)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid, FIMAQ_RUN_NAME=name, FIMAQ_CAPTURE_CALIBRATION_HASH="1")
            console = directory/f"seed{seed}_{arm}_console.log"
            started = time.perf_counter()
            print(shlex.join(cmd), flush=True)
            with console.open("w", encoding="utf-8") as handle:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        encoding="utf-8", errors="replace")
                try:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        handle.write(line)
                    code = proc.wait()
                except BaseException:
                    proc.terminate(); proc.wait()
                    raise
            if code: raise subprocess.CalledProcessError(code, cmd)
            run = validate_run(console, seed, arm, args)
            after, _ = source_fingerprint(args)
            if after != fingerprint:
                raise RuntimeError("Source/checkpoint changed during the run; do not pool this result")
            atomic_json(target, dict(status="completed", protocol=PROTOCOL, fingerprint=fingerprint,
                                     seed=seed, arm=arm, command=cmd, run=run,
                                     elapsed_seconds=time.perf_counter()-started,
                                     completed_at=datetime.now(timezone.utc).isoformat()))


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("seed3407", "extra-seeds", "all", "summary"), default="seed3407")
    parser.add_argument("--worker", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-slots", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--dataset", default="../imagenet_fimaq")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = get_parser(); args = parser.parse_args()
    try: jobs = planned_jobs(args.suite, args.worker, args.workers)
    except ValueError as e: parser.error(str(e))
    if args.num_workers < 0: parser.error("num-workers must be nonnegative")
    if args.dry_run and args.suite != "summary":
        for seed, arm in jobs:
            print(f"seed={seed} arm={arm}"); print(shlex.join(command(seed, arm, args)))
        return
    fingerprint, files = source_fingerprint(args)
    directory = ROOT/"checkpoints/qk_repair"/fingerprint[:16]
    if args.suite == "summary":
        summarize(directory, fingerprint, write=not args.dry_run)
        print(f"Result directory: {directory}")
        return
    if os.name != "posix": parser.error("GPU runs require Linux; use --dry-run on Windows")
    dataset = base.dataset_path(args.dataset)
    if not all((dataset/s).is_dir() for s in ("train", "val")): parser.error("Missing train/val dataset")
    entry = (ROOT/"test_quant.py").read_text(encoding="utf-8")
    core = (ROOT/"utils/qk_joint.py").read_text(encoding="utf-8")
    if "--qk-joint-reconstruction-loss" not in entry or "reconstruction_objective" not in core:
        parser.error("Sync test_quant.py and utils/qk_joint.py before running")
    atomic_json(directory/f"sources_worker{args.worker}.json", dict(fingerprint=fingerprint, files=files))
    for seed, arm in jobs:
        run_job(seed, arm, args, directory, fingerprint)
        summarize(directory, fingerprint)


if __name__ == "__main__": main()
