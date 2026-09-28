"""Read a completed Fisher probe run and print research-oriented aggregates.

This script is deliberately read-only: it does not load or modify the model.  It
combines the per-image JSONL records so that basis quality, projected/full KL,
finite-difference accuracy, timing, and incremental CUDA memory can be assessed.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def stats(values):
    values = [float(value) for value in values]
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.run_dir / "diagnostics.jsonl").read_text().splitlines()]
    events = defaultdict(list)
    for row in rows:
        events[row["event"]].append(row)

    output = {
        "event_counts": {name: len(items) for name, items in events.items()},
        "event_fields": {name: sorted(items[0]) for name, items in events.items()},
    }

    basis = {}
    for row in events["basis"]:
        basis[row["module"]] = {
            key: row.get(key)
            for key in (
                "retained_rank", "error_rank95", "retained_energy",
                "numerical_rank", "error_condition_positive", "singular_values",
            )
        }
    output["basis"] = basis

    quant = defaultdict(list)
    for row in events["quantization_error"]:
        quant[row["module"]].append(row)
    output["quantization_error"] = {}
    for module, items in quant.items():
        full = [row["actual_full_kl"] for row in items]
        projected = [row["actual_kl"] for row in items]
        coverage_key = next(
            (key for key in items[0] if "coverage" in key),
            None,
        )
        coverage = [row[coverage_key] for row in items] if coverage_key else []
        output["quantization_error"][module] = {
            "actual_full_kl": stats(full),
            "actual_projected_kl": stats(projected),
            "projected_to_full_kl_ratio_of_sums": sum(projected) / sum(full),
            "error_subspace_coverage": stats(coverage) if coverage else None,
            "available_fields": sorted(items[0]),
        }

    reference = defaultdict(list)
    for row in events["reference"]:
        reference[(row["module"], row["rank"])].append(row)
    forward = defaultdict(list)
    for row in events["forward_comparison"]:
        forward[(row["module"], row["rank"], row["epsilon"])].append(row)

    output["benchmarks"] = {}
    for (module, rank), refs in reference.items():
        ref_seconds = mean(row["seconds"] for row in refs)
        ref_memory = mean(row["incremental_peak_bytes"] for row in refs)
        record = {
            "reference_seconds": stats(row["seconds"] for row in refs),
            "reference_incremental_peak_mib": stats(row["incremental_peak_bytes"] / 2**20 for row in refs),
            "eval_error_projection_energy": stats(row["error_projection_energy"] for row in refs),
            "forward": {},
        }
        for (f_module, f_rank, epsilon), items in forward.items():
            if (f_module, f_rank) != (module, rank):
                continue
            seconds = mean(row["seconds"] for row in items)
            memory = mean(row["incremental_peak_bytes"] for row in items)
            record["forward"][str(epsilon)] = {
                "seconds": stats(row["seconds"] for row in items),
                "speedup_vs_reference": ref_seconds / seconds,
                "incremental_peak_mib": stats(row["incremental_peak_bytes"] / 2**20 for row in items),
                "incremental_memory_ratio_vs_reference": memory / ref_memory if ref_memory else None,
                "relative_fisher_error": stats(row["relative_fisher_error"] for row in items),
                "relative_response_error": stats(row["relative_response_error"] for row in items),
                "suffix_forward_evaluations": sorted(set(row["suffix_forward_evaluations"] for row in items)),
            }
        output["benchmarks"][f"{module}|r{rank}"] = record

    if args.compact:
        for module in basis:
            basis_row = basis[module]
            quant_row = output["quantization_error"][module]
            bench = output["benchmarks"][f"{module}|r8"]
            print(
                module,
                "basis_energy", f"{basis_row['retained_energy']:.4f}",
                "basis_r95", basis_row["error_rank95"],
                "eval_coverage", f"{bench['eval_error_projection_energy']['mean']:.4f}",
                "proj/full_KL", f"{quant_row['projected_to_full_kl_ratio_of_sums']:.4f}",
                "ref_s", f"{bench['reference_seconds']['mean']:.4f}",
                "ref_mem_MiB", f"{bench['reference_incremental_peak_mib']['mean']:.2f}",
            )
            for epsilon, result in bench["forward"].items():
                print(
                    "  eps", epsilon,
                    "fisher_err", f"{result['relative_fisher_error']['mean']:.6f}",
                    "response_err", f"{result['relative_response_error']['mean']:.6f}",
                    "speedup", f"{result['speedup_vs_reference']:.2f}x",
                    "mem_MiB", f"{result['incremental_peak_mib']['mean']:.2f}",
                    "forwards", result["suffix_forward_evaluations"],
                )
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
