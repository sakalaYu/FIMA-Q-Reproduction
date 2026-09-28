"""Summarize diagnostic JSONL into CSV; optionally plot projected spectra.

Usage: python scripts/summarize_fisher_probe.py PATH_TO_RESULT_DIRECTORY [--plot]
"""
import argparse
import csv
import json
import pathlib
from collections import defaultdict
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=pathlib.Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.run/'diagnostics.jsonl').read_text().splitlines() if s.strip()]
    complete = (args.run/'complete.json').exists()
    print('Complete:', complete)
    groups = defaultdict(list)
    for row in rows:
        if row['event'] in ('spectrum', 'forward_comparison', 'reference'):
            groups[(row['event'], row['module'], row['rank'], row.get('epsilon',''))].append(row)
    summaries = []
    keys = ('rank95','trace','projected_offdiag_ratio','relative_fisher_error',
            'relative_response_error','seconds','peak_allocated_bytes','error_projection_energy')
    for (event,module,rank,epsilon), group in groups.items():
        result = dict(event=event,module=module,rank=rank,epsilon=epsilon,n=len(group),complete=complete)
        for key in keys:
            values = [r[key] for r in group if r.get(key) is not None]
            result[key] = mean(values) if values else ''
        summaries.append(result)
    if summaries:
        with open(args.run/'summary.csv','w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    # Aggregate squared prediction error rather than dividing by tiny per-image KL.
    errors = defaultdict(list)
    for row in rows:
        if row['event'] in ('heldout_direction', 'quantization_error'):
            for method,predictions in row['predictions'].items():
                for approximation,pred in predictions.items():
                    errors[(row['event'],row['module'],row['rank'],method,approximation)].append((pred,row['actual_kl']))
    with open(args.run/'prediction_errors.csv','w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['event','module','rank','method','approximation','n','relative_rmse','mean_true_kl'])
        for key,pairs in errors.items():
            error = (sum((p-y)**2 for p,y in pairs)/max(sum(y*y for p,y in pairs),1e-30))**0.5
            writer.writerow([*key,len(pairs),error,mean(y for p,y in pairs)])
    if args.plot:
        try:
            import matplotlib
        except ModuleNotFoundError:
            print('matplotlib is not installed; CSV summaries were written, skipping optional PNG plots.')
            return
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for (event,module,rank,epsilon),group in groups.items():
            if event != 'spectrum': continue
            values = [mean(r['eigenvalues'][j] for r in group) for j in range(rank)]
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(range(1,rank+1),values,marker='o')
            ax.set(xlabel='Eigenvalue index',ylabel='Projected Fisher eigenvalue',
                   title=module+' (subspace dimension '+str(rank)+')')
            fig.tight_layout()
            fig.savefig(args.run/(module+'_rank'+str(rank)+'.png'),dpi=160)
            plt.close(fig)
    print('Wrote summary.csv and prediction_errors.csv to',args.run)


if __name__ == '__main__':
    main()
