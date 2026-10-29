"""Matched Q/K loss/rotation controls and the preceding low-dimensional trial."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import run_qk_joint_experiments as qk


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    label: str
    variant: str
    loss: str
    protocol: str = qk.PROTOCOL
    rank: int = 0
    gate: float = 0.0


# This order distributes the expensive DPLR trials across three windows.
JOBS = (
    Job("C_dplr", "C", "dplr"),
    Job("B_dplr", "B", "dplr"),
    Job("C_mse", "C", "mse"),
    Job("C_lowdim", "C", "dplr", "v3_lowrank_gate", 4, 0.0),
    Job("B_mse", "B", "mse"),
    Job("A_dplr", "A", "dplr"),
    Job("C_lowdim_gate", "C", "dplr", "v3_lowrank_gate", 4, 0.05),
)
SUITES = {
    "causal": ("C_dplr", "B_dplr", "C_mse", "B_mse"),
    "aligned": ("A_dplr", "B_dplr", "C_dplr"),
    "rotation": ("C_dplr", "C_lowdim", "C_lowdim_gate"),
    "all": tuple(job.label for job in JOBS),
}


def job_flags(job):
    return ["--variant", job.variant, "--reconstruction-loss", job.loss,
            "--protocol", job.protocol, "--rotation-rank", str(job.rank),
            "--min-relative-kl", str(job.gate)]


def job_args(job, args):
    return qk.get_parser().parse_args(job_flags(job) + [
        "--seed", str(args.seed), "--dataset", args.dataset,
        "--num-workers", str(args.num_workers), "--gpu", args.gpu,
        "--gpu-slots", str(args.gpu_slots)])


def selected_jobs(suite, worker=1, workers=1):
    selected = [job for job in JOBS if job.label in SUITES[suite]]
    return selected[worker - 1::workers]


def matched_inputs(left, right):
    return all(left[key] == right[key] for key in ("seed", "bits", "cfg", "calibration_hashes")) and (
        left["args"].get("num_workers") == right["args"].get("num_workers")) and (
        qk.base.dataset_path(left["args"]["dataset"]) ==
        qk.base.dataset_path(right["args"]["dataset"]))


def summarize(args, write=True):
    runs = qk.available_runs()
    found = {}
    rows = []
    for job in JOBS:
        run = qk.latest(runs, job.variant, job_args(job, args), protocol=job.protocol)
        if run is None:
            print(f"{job.label}: incomplete")
            continue
        found[job.label] = run
        rows.append(dict(label=job.label, variant=job.variant, loss=job.loss,
                         protocol=job.protocol, rotation_rank=job.rank, gate=job.gate, run=run))
        print(f"{job.label}: Top-1 {run['top1']:.3f}; accepted "
              f"{sum(block['accepted'] for block in run['qk_blocks'])}/12")

    contrasts = []

    def difference(label, left, right):
        if left not in found or right not in found:
            return None
        valid = matched_inputs(found[left], found[right])
        delta = found[left]["top1"] - found[right]["top1"] if valid else None
        contrasts.append(dict(label=label, left=left, right=right,
                              matched_inputs=valid, delta_top1_pp=delta))
        print(f"{label}: {delta:+.3f} pp" if valid else f"{label}: input mismatch; excluded")
        return delta

    rotation_mse = difference("rotation_at_mse", "C_mse", "B_mse")
    rotation_dplr = difference("rotation_at_dplr", "C_dplr", "B_dplr")
    difference("loss_without_rotation", "B_dplr", "B_mse")
    difference("loss_with_rotation", "C_dplr", "C_mse")
    difference("dual_scaling_at_dplr", "B_dplr", "A_dplr")
    difference("lowdim_parameterization", "C_lowdim", "C_dplr")
    difference("five_percent_gate", "C_lowdim_gate", "C_lowdim")
    interaction = None
    if rotation_mse is not None and rotation_dplr is not None:
        four = [found[key] for key in ("B_mse", "B_dplr", "C_mse", "C_dplr")]
        if all(matched_inputs(four[0], run) for run in four[1:]):
            interaction = rotation_dplr - rotation_mse
            print(f"rotation_loss_interaction: {interaction:+.3f} pp")

    fixed = [run for run in qk.base.available_runs()
             if run["seed"] == args.seed and run["bits"] == 3 and run["rotation_mode"] == "all"
             and run["python_seeded"] and run["args"].get("num_workers") == args.num_workers
             and qk.base.dataset_path(run["args"]["dataset"]) == qk.base.dataset_path(args.dataset)]
    for reference in fixed:
        print(f"Fixed control: {reference['top1']:.3f}; {reference['run_id']}")
    for row in rows:
        row["fixed_comparisons"] = [dict(run_id=reference["run_id"], top1=reference["top1"],
                                        delta_top1_pp=row["run"]["top1"] - reference["top1"])
                                    for reference in fixed if matched_inputs(row["run"], reference)]
    fix_comparison = None
    historical_c = [run for run in runs if run["args"].get("qk_joint_protocol", "v1") == "v1"]
    old_c = qk.latest(historical_c, "C", job_args(JOBS[0], args))
    if "C_dplr" in found and old_c is not None and matched_inputs(found["C_dplr"], old_c):
        fix_comparison = dict(old_run=old_c["run_id"], new_run=found["C_dplr"]["run_id"],
                              delta_top1_pp=found["C_dplr"]["top1"] - old_c["top1"])
        print(f"C_v2_vs_v1: {fix_comparison['delta_top1_pp']:+.3f} pp")
    report = dict(seed=args.seed, rows=rows, complete_count=len(rows), expected_count=len(JOBS),
                  contrasts=contrasts, rotation_loss_interaction_pp=interaction,
                  fixed_controls=fixed, c_v2_vs_v1=fix_comparison)
    if write:
        directory = ROOT / "checkpoints/qk_joint"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"controlled_summary_seed{args.seed}.json"
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(target)
        print(f"Summary: {target}")
    return report


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="all")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--worker", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset", default="../imagenet_fimaq")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-slots", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not 1 <= args.worker <= args.workers or args.num_workers < 0:
        parser.error("Require 1 <= worker <= workers and num-workers >= 0")
    if args.summary:
        summarize(args, write=not args.dry_run)
        return
    for job in selected_jobs(args.suite, args.worker, args.workers):
        options = job_args(job, args)
        print(f"Worker {args.worker}/{args.workers}: {job.label}", flush=True)
        if args.dry_run:
            print(shlex.join(qk.command(job.variant, options)))
            continue
        command = [sys.executable, "-u", str(ROOT / "scripts/run_qk_joint_experiments.py")]
        command += job_flags(job) + ["--seed", str(args.seed), "--dataset", args.dataset,
                                   "--num-workers", str(args.num_workers), "--gpu", args.gpu,
                                   "--gpu-slots", str(args.gpu_slots)]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        summarize(args)


if __name__ == "__main__":
    main()
