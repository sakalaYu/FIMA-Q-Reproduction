"""One decisive W3A3 trial of signed task-aligned attention relations."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

import run_headwise_experiments as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "relation_v2_alignment"
DEFAULTS = dict(relation_weight=0.1, relation_queries=16,
                relation_start=0.2, relation_full=0.6)


def run_name(seed):
    return f"headwise_alignment_w3a3_seed{seed}_seeded_full_{PROTOCOL}"


def command(args):
    result = base.command(3, args.seed, True, args.dataset, args.num_workers)
    result.extend(["--attention-relation", "aligned", "--relation-protocol", PROTOCOL])
    for key in DEFAULTS:
        result.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    return result


def completed_runs(args):
    paths = list((ROOT / "logs").glob("*headwise_alignment*.log"))
    paths += list((ROOT / "checkpoints/quant_result").glob("*headwise_alignment*/output.log"))
    found = {}
    for path in paths:
        run = base.read_run(path)
        if run is None or run["seed"] != args.seed or run["bits"] != 3:
            continue
        if run["rotation_mode"] != "all" or not run["python_seeded"]:
            continue
        options = run["args"]
        if options.get("attention_relation") != "aligned" or options.get("relation_protocol") != PROTOCOL:
            continue
        if options.get("num_workers") != args.num_workers:
            continue
        if base.dataset_path(options["dataset"]) != base.dataset_path(args.dataset):
            continue
        if any(options.get(key) != getattr(args, key) for key in DEFAULTS):
            continue
        if [part[0] for part in run["calibration_hashes"]] != ["128", "1024"]:
            continue
        detail = ROOT / "checkpoints/quant_result" / run["run_id"] / "attention_relation.json"
        if not detail.is_file():
            continue
        try:
            report = json.loads(detail.read_text(encoding="utf-8"))
            blocks = report["blocks"]
            if report["options"].get("protocol") != PROTOCOL:
                continue
            if len(blocks) != 12 or {block.get("block") for block in blocks} != {
                    f"blocks.{i}" for i in range(12)}:
                continue
            if any(block.get("protocol") != PROTOCOL or block.get("mode") != "aligned"
                   for block in blocks):
                continue
            run["relation_blocks"] = blocks
        except (OSError, ValueError, TypeError, KeyError):
            continue
        found[run["run_id"]] = run
    return sorted(found.values(), key=lambda run: run["run_id"])


def latest_reference(glob, args):
    paths = list((ROOT / "logs").glob(glob + ".log"))
    paths += list((ROOT / "checkpoints/quant_result").glob(glob + "/output.log"))
    runs = []
    for path in paths:
        run = base.read_run(path)
        if run is None or run["seed"] != args.seed or run["bits"] != 3:
            continue
        if run["rotation_mode"] != "all" or not run["python_seeded"]:
            continue
        if base.dataset_path(run["args"]["dataset"]) != base.dataset_path(args.dataset):
            continue
        if run["args"].get("num_workers") != args.num_workers:
            continue
        if [part[0] for part in run["calibration_hashes"]] != ["128", "1024"]:
            continue
        runs.append(run)
    return sorted(runs, key=lambda run: run["run_id"])[-1] if runs else None


def summarize(args, write=True):
    runs = completed_runs(args)
    current = runs[-1] if runs else None
    control = latest_reference("*headwise_relation*control*", args)
    old_fixed = latest_reference("*headwise_w3a3*rotation*seeded_full*", args)
    references = {}
    for label, reference in (("fresh_control", control), ("prior_fixed", old_fixed)):
        matched = current is not None and reference is not None and all(
            current[key] == reference[key] for key in ("cfg", "calibration_hashes"))
        references[label] = dict(run=reference, matched=bool(matched),
                                 delta_top1_pp=current["top1"] - reference["top1"] if matched else None)
    report = dict(protocol=PROTOCOL, seed=args.seed, run=current, references=references)
    if current is None:
        print("aligned: incomplete")
    else:
        blocks = current["relation_blocks"]
        active = sum(block["active_steps"] > 0 for block in blocks)
        fractions = [block["alignment_positive_fraction"] for block in blocks]
        print(f"aligned: Top-1 {current['top1']:.3f}; Top-5 {current['top5']:.3f}; "
              f"loss {current['loss']:.3f}; active blocks {active}/12")
        print(f"  final positive head/query fraction: {sum(fractions) / len(fractions):.3f} mean")
        for label, item in references.items():
            delta = item["delta_top1_pp"]
            print(f"  vs {label}: {delta:+.3f} pp" if delta is not None
                  else f"  vs {label}: N/A (reference/config/hash mismatch)")
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
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset", default="../imagenet_fimaq")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-slots", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.seed < 0 or args.num_workers < 0 or args.relation_weight <= 0 or args.relation_queries < 1:
        parser.error("seed/workers must be nonnegative; relation weight/queries positive")
    if not 0 <= args.relation_start < args.relation_full <= 1:
        parser.error("Require 0 <= relation-start < relation-full <= 1")
    if args.summary:
        summarize(args, write=not args.dry_run)
        return
    cmd = command(args)
    name = run_name(args.seed)
    if args.dry_run:
        print(f"{name}; GPU {args.gpu}, slots={args.gpu_slots}")
        print(shlex.join(cmd))
        return
    if os.name != "posix":
        parser.error("Actual runs require Linux; use --dry-run on Windows")
    dataset = base.dataset_path(args.dataset)
    if not all((dataset / split).is_dir() for split in ("train", "val")):
        parser.error(f"Missing train/val directories under {dataset}")
    checks = (("test_quant.py", "relation_v2_alignment"),
              ("utils/attention_relation.py", "refresh_score_forward"),
              ("utils/block_recon.py", "refresh_score_forward"),
              ("utils/datasets.py", "FIMAQ_CAPTURE_CALIBRATION_HASH"))
    for relative, marker in checks:
        source = ROOT / relative
        if not source.is_file() or marker not in source.read_text(encoding="utf-8"):
            parser.error(f"Sync {relative} before launching")
    with base.job_lock(name):
        if args.resume and completed_runs(args):
            print(f"Skipping completed run: {completed_runs(args)[-1]['path']}")
            summarize(args)
            return
        with base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
            before = {run["run_id"] for run in completed_runs(args)}
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
        fresh = [run for run in completed_runs(args) if run["run_id"] not in before]
        if not fresh:
            raise RuntimeError("No newly completed aligned relation run with 12 block reports")
    summarize(args)


if __name__ == "__main__":
    main()
