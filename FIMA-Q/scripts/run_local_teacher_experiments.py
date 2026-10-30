"""Three tmux workers sharing one GPU; settings live in a JSON, not bash."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from run_local_teacher import (ROOT, DEFAULT_CONFIG, read_config, resolve_path,
                               experiment_identity, atomic_json)

SUITES = {
    "prepare": ["prepare"], "probe": ["probe"],
    "screen": ["E3", "E4", "E5"],
    "controls": ["E1", "E2", "E6"],
    "all": ["E1", "E2", "E3", "E4", "E5", "E6"],
}


def planned_jobs(suite, worker, workers):
    if not 1 <= worker <= workers <= 3:
        raise ValueError("Require 1 <= worker <= workers <= 3")
    return SUITES[suite][worker - 1::workers]


def command(config, arm, signature):
    return [sys.executable, "-u", "scripts/run_local_teacher.py", "--config", str(config),
            "--arm", arm, "--signature", signature, "--device", "cuda:0"]


def summary(directory):
    results = {}
    for arm in SUITES["all"]:
        path = directory / arm / "result.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("status") == "completed":
                results[arm] = value
                print("%s: %.3f%%, delta %+.3f pp, repaired=%d, damaged=%d, %.1f min" %
                      (arm, value["top1"], value["delta_pp"], value["repaired"],
                       value["damaged"], value["elapsed_seconds"] / 60))
    contrasts = {}
    for name, left, right in (("quant_suffix_minus_fp", "E5", "E4"),
                              ("global_anchor_increment", "E6", "E5"),
                              ("complete_minus_standard_kd", "E6", "E3")):
        if left in results and right in results:
            contrasts[name] = results[left]["top1"] - results[right]["top1"]
    value = dict(results=results, contrasts_pp=contrasts)
    atomic_json(directory / "summary.json", value)
    print("Contrasts:", contrasts)
    print("Results:", directory)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", choices=list(SUITES) + ["summary"], default="screen")
    parser.add_argument("--worker", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-slots", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = read_config(config_path)
    try:
        jobs = planned_jobs(args.suite, args.worker, args.workers) if args.suite != "summary" else []
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        for arm in jobs:
            print("%s, GPU=%s, slots=%s" % (arm, args.gpu, args.gpu_slots))
            print(shlex.join(command(config_path, arm, "<computed-after-checkpoint-check>")))
        return
    signature, identity = experiment_identity(cfg)
    directory = resolve_path(cfg["output_root"]) / signature[:16]
    if args.suite == "summary":
        summary(directory)
        return
    if os.name != "posix":
        parser.error("GPU launcher requires Linux; --dry-run is available on Windows")
    from run_headwise_experiments import gpu_lock, job_lock
    directory.mkdir(parents=True, exist_ok=True)
    if args.suite != "prepare" and not (directory / "prepared.json").exists():
        parser.error("Run --suite prepare first with this same config")
    atomic_json(directory / ("sources_worker%d.json" % args.worker), identity)
    for arm in jobs:
        result_path = directory / arm / ("probe.json" if arm == "probe" else "result.json")
        with job_lock("local_teacher_%s_%s" % (signature[:16], arm)):
            if args.resume and result_path.exists():
                if arm != "probe" and json.loads(result_path.read_text()).get("status") != "completed":
                    raise RuntimeError("Result is marked invalid; inspect it before rerunning")
                print("Skip completed:", arm, flush=True)
                continue
            if (directory / arm / "progress.pt").exists() and not args.resume:
                parser.error("Progress exists for %s; pass --resume or use a new JSON configuration" % arm)
            if result_path.exists():
                parser.error("Completed result exists; pass --resume to reuse it")
            with gpu_lock(args.gpu, args.gpu_slots) as uuid:
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = uuid
                environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                target = directory / (arm + "_console.log")
                cmd = command(config_path, arm, signature)
                print(shlex.join(cmd), flush=True)
                with target.open("a", encoding="utf-8") as log:
                    log.write("\nLAUNCH %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
                    log.flush()
                    process = subprocess.Popen(cmd, cwd=ROOT, env=environment,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace", bufsize=1)
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        status = process.wait()
                    except BaseException:
                        process.terminate()
                        process.wait()
                        raise
                if status:
                    raise subprocess.CalledProcessError(status, cmd)
                after, _ = experiment_identity(cfg)
                if after != signature:
                    if result_path.exists() and arm not in ("prepare", "probe"):
                        invalid = json.loads(result_path.read_text())
                        invalid["status"] = "invalid_source_change"
                        atomic_json(result_path, invalid)
                    raise RuntimeError("Sources/checkpoint changed during run; exclude this result")
        if arm not in ("prepare", "probe"):
            summary(directory)


if __name__ == "__main__":
    main()
