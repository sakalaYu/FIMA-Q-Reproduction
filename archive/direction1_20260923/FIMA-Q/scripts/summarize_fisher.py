"""Print actual Fisher calls, probe overhead and drift from diagnostic JSONL.

Usage: python scripts/summarize_fisher.py checkpoints/quant_result/RUN/fisher_diagnostics.jsonl
"""
import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    args = parser.parse_args()
    blocks = defaultdict(list)
    with open(args.log, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                blocks[row['block']].append(row)
    print('block\tupdates\tprobes\ttriggers\tmax_rel_error\tfisher_s\tprobe_s\tcomplete')
    for block, rows in blocks.items():
        probes = [r for r in rows if r['event'] == 'probe']
        summaries = [r for r in rows if r['event'] == 'summary']
        summary = summaries[-1] if summaries else {}
        values = [block, summary.get('updates', sum(r['event'] == 'refresh' for r in rows)),
                  summary.get('probe_calls', len(probes)), sum(r['trigger'] for r in probes),
                  round(max((r['relative_error'] for r in probes), default=0), 4),
                  round(summary.get('fisher_seconds', 0), 2),
                  round(summary.get('probe_seconds', 0), 2), bool(summaries)]
        print('\t'.join(map(str, values)))


if __name__ == '__main__':
    main()
