"""Read-only analysis of direction-1 run artifacts; prints summaries as JSON."""
import json
import pathlib
from collections import Counter
from datetime import datetime

root = pathlib.Path(__file__).resolve().parents[1] / 'FIMA-Q'
for run in sorted((root / 'checkpoints/quant_result').glob('*direction1*')):
    print('RUN', run.name)
    for name in ('experiment.json', 'probe_split.json'):
        p = run / name
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            if name == 'probe_split.json':
                data = {k: (dict(n=len(v), first=v[:3], last=v[-3:]) if isinstance(v, list) else v) for k,v in data.items()}
            if name == 'experiment.json':
                print('CONFIG', json.dumps(data))
    lines=(run/'output.log').read_text(encoding='utf-8',errors='replace').splitlines()
    print('LOG_SUMMARY', '\n'.join(s for s in lines if any(k in s for k in ('start the process','finished the process','guided block reconstruction','Validating',' * Prec@','Fisher split','Traceback','Error','Saving checkpoint'))))
    p=run/'fisher_diagnostics.jsonl'
    if not p.exists(): continue
    rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    print('EVENTS',Counter(r['event'] for r in rows))
    for block in dict.fromkeys(r['block'] for r in rows):
        br=[r for r in rows if r['block']==block]
        probes=[r for r in br if r['event']=='probe']
        refresh=[r for r in br if r['event']=='refresh']
        summary=[r for r in br if r['event']=='summary']
        print('BLOCK',json.dumps(dict(block=block,updates=summary[-1]['updates'],
            max_excess=round(max(r['relative_error']-r['anchor_error'] for r in probes),5),
            trend_steps=[r['step'] for r in probes if r['trend_bad']],
            initial_kl=refresh[0]['kl_mean'],initial_scale=refresh[0]['scale'])))
    sums=[r for r in rows if r['event']=='summary']
    print('TOTAL',json.dumps({key:sum(r[key] for r in sums) for key in ('updates','probe_calls','probe_seconds','fisher_seconds')}))
    print('BAD_TOTAL', {key:sum(r[key] for r in rows if r['event']=='probe') for key in ('error_bad','trend_bad','trigger')})

runs=sorted((root/'checkpoints/quant_result').glob('*direction1*'))
a,b=[json.loads((r/'experiment.json').read_text()) for r in runs]
print('ARGS_DIFF', {k:[v,b['args'].get(k)] for k,v in a['args'].items() if v!=b['args'].get(k)})
print('CFG_IDENTICAL',a['config']==b['config'])
print('SPLIT_IDENTICAL',(runs[0]/'probe_split.json').read_bytes()==(runs[1]/'probe_split.json').read_bytes())
for path,terms in [('utils/block_recon.py',['probe_state','quantized=True','cur_inp','fork_rng']),('test_quant.py',['seed_all','manual_seed','random.seed']),('utils/datasets.py',['train_transform','preloaded_data','is_training=True'])]:
    print('CODE',path)
    for i,line in enumerate((root/path).read_text(encoding='utf-8').splitlines(),1):
        if any(t in line for t in terms): print(i,line)
