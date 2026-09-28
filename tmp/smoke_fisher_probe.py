"""Local integration check using synthetic images, never an accuracy experiment.
Loads the real calibrated checkpoint and runs both diagnostic paths on CPU.
"""
import pathlib
import sys
root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tmp/probe_test_dependencies'))
sys.path.insert(0,str(root/'FIMA-Q/scripts'))
from PIL import Image
import numpy as np
import torch
torch.set_num_threads(2)
folder=root/'tmp/probe_test_data/train/synthetic'
folder.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(1)
for i in range(3):
    Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8)).save(folder/(str(i)+'.png'))
from probe_fisher import main, parser
args=parser().parse_args([
    '--dataset',str(root/'tmp/probe_test_data'), '--cache',str(root/'tmp/probe_test_data/cache.pt'),
    '--output-root',str(root/'tmp/probe_test_results'), '--basis-size','2','--eval-size','1',
    '--ranks','2','--blocks','0','11','--epsilons','0.001','--test-directions','1','--device','cpu'])
main(args)
