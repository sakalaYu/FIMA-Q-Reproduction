"""Archive only method-1 artifacts; restore its two integration files from Git.
No history reset, no model/data deletion. All moves are checked inside workspace.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
archive = root / 'archive' / 'direction1_20260923'
if archive.exists():
    raise SystemExit('Archive exists; refusing to overwrite')
restore = ['FIMA-Q/test_quant.py', 'FIMA-Q/utils/block_recon.py']
# Obtain and validate originals before moving anything.
originals = {p: subprocess.check_output(['git', 'show', '4d5e8ea:' + p], cwd=root) for p in restore}
paths = [root / p for p in [
    'FIMA-Q/utils/fisher_reliability.py', 'FIMA-Q/scripts/run_direction1.sh',
    'FIMA-Q/scripts/summarize_fisher.py', 'FIMA-Q/DIRECTION1.md',
    'FIMA-Q/DIRECTION1_RESULTS_20260923.md', 'tmp/analyze_direction1.py',
    'FIMA-Q/tests/test_fisher_probe.py', 'FIMA-Q/tests/test_fisher_reliability.py',
    'FIMA-Q/tests/test_fisher_schedule.py']]
paths += list((root/'FIMA-Q/checkpoints/quant_result').glob('*_direction1_*'))
paths += list((root/'FIMA-Q/logs').glob('*_direction1_*.log'))
paths = [p for p in paths if p.exists()]
for p in paths + [archive]:
    p.resolve().relative_to(root.resolve())
records = []
for p in paths + [root/p for p in restore]:
    for f in ([p] if p.is_file() else p.rglob('*')):
        if f.is_file():
            records.append(dict(path=f.relative_to(root).as_posix(),
                                sha256=hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()
                                if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read_bytes()).hexdigest()))
archive.mkdir(parents=True)
for rel in restore:
    dst = archive/rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root/rel, dst)
for p in paths:
    dst = archive/p.relative_to(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(dst))
for rel, data in originals.items():
    (root/rel).write_bytes(data)
(archive/'manifest.json').write_text(json.dumps(dict(restored_from='4d5e8ea', files=records), indent=2), encoding='utf-8')
print('Archived', len(records), 'files to', archive)
print('Restored baseline integration files from 4d5e8ea; Git history unchanged.')
