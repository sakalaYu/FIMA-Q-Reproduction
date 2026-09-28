"""Summarize a fixed-step directional Fisher diagnostic result directory."""
import argparse
import csv
import json
import pathlib
from collections import defaultdict
from statistics import mean


def relative_rmse(pairs):
    return (sum((prediction - target) ** 2 for prediction, target in pairs) /
            max(sum(target ** 2 for _, target in pairs), 1e-30)) ** 0.5


def optional_mean(values, scale=1):
    values = [value / scale for value in values if value is not None]
    return mean(values) if values else ''


def pearson(pairs):
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    xbar, ybar = mean(xs), mean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in pairs)
    denominator = (sum((x - xbar) ** 2 for x in xs) *
                   sum((y - ybar) ** 2 for y in ys)) ** 0.5
    return numerator / denominator if denominator else ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=pathlib.Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'diagnostics.jsonl').read_text().splitlines() if line.strip()]
    complete = (args.run / 'complete.json').exists()

    references, forwards, predictions = defaultdict(list), defaultdict(list), defaultdict(list)
    amplitudes = defaultdict(list)
    for row in rows:
        if row['event'] == 'reference':
            key = (row['module'], row['variant'], row['rank'])
            references[key].append(row)
        elif row['event'] == 'forward_comparison':
            key = (row['module'], row['variant'], row['rank'])
            forwards[key].append(row)
        elif row['event'] == 'directional_prediction':
            key = (row['module'], row['variant'], row['rank'])
            for method, approximations in row['predictions'].items():
                for approximation, prediction in approximations.items():
                    predictions[key + (method, approximation)].append((prediction, row['actual_kl']))
        elif row['event'] == 'amplitude_sweep':
            for method, prediction in row['predictions'].items():
                pair = (prediction, row['actual_kl'])
                amplitudes[(row['module'], method, row['scale'])].append(pair)
                if abs(row['scale'] - 1.) > 1e-12:
                    amplitudes[(row['module'], method, 'all_non_anchor')].append(pair)

    fields = ['module', 'variant', 'rank', 'n', 'epsilon', 'complete',
              'reference_seconds', 'forward_seconds', 'speedup',
              'reference_incremental_peak_mib', 'forward_incremental_peak_mib',
              'relative_fisher_error', 'relative_response_error']
    with open(args.run / 'summary.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(forwards):
            ref, fwd = references[key], forwards[key]
            ref_seconds, fwd_seconds = mean(x['seconds'] for x in ref), mean(x['seconds'] for x in fwd)
            writer.writerow(dict(
                module=key[0], variant=key[1], rank=key[2], n=len(fwd), epsilon=fwd[0]['epsilon'],
                complete=complete, reference_seconds=ref_seconds, forward_seconds=fwd_seconds,
                speedup=ref_seconds / fwd_seconds,
                reference_incremental_peak_mib=optional_mean(
                    (x['incremental_peak_bytes'] for x in ref), 2**20),
                forward_incremental_peak_mib=optional_mean(
                    (x['incremental_peak_bytes'] for x in fwd), 2**20),
                relative_fisher_error=mean(x['relative_fisher_error'] for x in fwd),
                relative_response_error=mean(x['relative_response_error'] for x in fwd),
            ))

    with open(args.run / 'prediction_errors.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['module', 'variant', 'rank', 'method', 'approximation', 'n',
                         'relative_rmse', 'mean_true_kl'])
        for key in sorted(predictions):
            pairs = predictions[key]
            writer.writerow([*key, len(pairs), relative_rmse(pairs), mean(target for _, target in pairs)])
    if amplitudes:
        with open(args.run / 'amplitude_errors.csv', 'w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['module', 'method', 'scale', 'n', 'relative_rmse',
                             'pearson', 'prediction_to_true_sum'])
            for key in sorted(amplitudes, key=lambda item: (item[0], item[1], str(item[2]))):
                pairs = amplitudes[key]
                writer.writerow([*key, len(pairs), relative_rmse(pairs), pearson(pairs),
                                 sum(prediction for prediction, _ in pairs) /
                                 max(sum(target for _, target in pairs), 1e-30)])
    print('Complete:', complete)
    print('Wrote CSV summaries to', args.run)


if __name__ == '__main__':
    main()
