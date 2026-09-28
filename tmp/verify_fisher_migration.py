"""Verify archived hashes and exact baseline restoration; no files are edited."""
import hashlib
import json
import pathlib
import subprocess
root=pathlib.Path(__file__).resolve().parents[1]
archive=root/'archive/direction1_20260923'
manifest=json.loads((archive/'manifest.json').read_text())
for item in manifest['files']:
    path=archive/item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], str(path)
for name in ('FIMA-Q/test_quant.py','FIMA-Q/utils/block_recon.py'):
    assert (root/name).read_bytes()==subprocess.check_output(['git','show','4d5e8ea:'+name],cwd=root)
print('PASS: archived hashes match for',len(manifest['files']),'files; both baseline integration files restored exactly.')
