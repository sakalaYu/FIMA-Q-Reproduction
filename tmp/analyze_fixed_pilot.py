"""Read-only analysis of a fixed forward-Fisher result directory."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def correlation(xs, ys):
    xbar, ybar = mean(xs), mean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - xbar) ** 2 for x in xs) * sum((y - ybar) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'diagnostics.jsonl').read_text().splitlines()]

    directions = [row for row in rows if row['event'] == 'direction_set']
    print('direction_reconstruction_max', max(row['reconstruction_relative_error'] for row in directions))

    timing = defaultdict(dict)
    for event in ('reference', 'forward_comparison'):
        groups = defaultdict(list)
        for row in rows:
            if row['event'] == event:
                groups[(row['module'], row['variant'])].append(row)
        for key, items in groups.items():
            timing[key][event] = items

    print('\nTIMING_AND_ESTIMATOR')
    for (module, variant), events in sorted(timing.items()):
        refs, fwds = events['reference'], events['forward_comparison']
        ref_time, fwd_time = mean(x['seconds'] for x in refs), mean(x['seconds'] for x in fwds)
        fisher_errors = [x['relative_fisher_error'] for x in fwds]
        response_errors = [x['relative_response_error'] for x in fwds]
        print(module, variant,
              'speedup', round(ref_time / fwd_time, 4),
              'fisher_mean/p95/max', *(round(x, 7) for x in
                  (mean(fisher_errors), percentile(fisher_errors, .95), max(fisher_errors))),
              'response_mean/p95/max', *(round(x, 7) for x in
                  (mean(response_errors), percentile(response_errors, .95), max(response_errors))))
    worst = max((row for row in rows if row['event'] == 'forward_comparison'),
                key=lambda row: row['relative_fisher_error'])
    print('worst_forward_row', worst)

    predictions = {}
    for row in rows:
        if row['event'] != 'directional_prediction':
            continue
        key = (row['module'], row['image_index'], row['variant'])
        predictions[key] = dict(actual=row['actual_kl'], **{
            method + '_' + approximation: value
            for method, approximations in row['predictions'].items()
            for approximation, value in approximations.items()
        })

    print('\nPREDICTION')
    modules = sorted(set(key[0] for key in predictions))
    for module in modules:
        items = [value for (name, _, variant), value in predictions.items()
                 if name == module and variant == 'single']
        actual = [x['actual'] for x in items]
        predicted = [x['finite_difference_full'] for x in items]
        ratios = [p / y for p, y in zip(predicted, actual) if y > 0]
        relative_errors = [abs(p - y) / y for p, y in zip(predicted, actual) if y > 0]
        print(module,
              'pred/true_sum', round(sum(predicted) / sum(actual), 5),
              'ratio_median', round(median(ratios), 5),
              'ratio_p05/p95', round(percentile(ratios, .05), 5), round(percentile(ratios, .95), 5),
              'per_image_rel_median/p95/max', *(round(x, 5) for x in
                  (median(relative_errors), percentile(relative_errors, .95), max(relative_errors))),
              'pearson', None if correlation(predicted, actual) is None else round(correlation(predicted, actual), 5))

    print('\nSINGLE_VS_GROUPED_FULL')
    for method in ('autograd', 'finite_difference'):
        full_differences = []
        for module in modules:
            for image_index in range(32):
                single = predictions[(module, image_index, 'single')][method + '_full']
                grouped = predictions[(module, image_index, 'grouped')][method + '_full']
                full_differences.append(abs(single - grouped) / max(abs(single), 1e-30))
        print(method, 'mean/p95/max', mean(full_differences),
              percentile(full_differences, .95), max(full_differences))

    for variant in ('single', 'grouped'):
        reference_total = sum(mean(x['seconds'] for x in events['reference'])
                              for key, events in timing.items() if key[1] == variant)
        forward_total = sum(mean(x['seconds'] for x in events['forward_comparison'])
                            for key, events in timing.items() if key[1] == variant)
        print('TOTAL', variant, 'reference_seconds', reference_total,
              'forward_seconds', forward_total, 'speedup', reference_total / forward_total)


if __name__ == '__main__':
    main()
