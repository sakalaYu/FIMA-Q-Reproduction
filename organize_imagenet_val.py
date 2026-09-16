import os
import shutil
from scipy.io import loadmat

VAL_DIR = "/root/autodl-tmp/imagenet_fimaq/val"
META_PATH = "/root/autodl-tmp/imagenet_devkit/ILSVRC2012_devkit_t12/data/meta.mat"
GT_PATH = "/root/autodl-tmp/imagenet_devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"

print("Loading ImageNet metadata...")

meta = loadmat(META_PATH, squeeze_me=True)
synsets = meta["synsets"]

id_to_wnid = {}

for synset in synsets:
    ilsvrc_id = int(synset[0])
    wnid = str(synset[1])
    num_children = int(synset[4])
    if num_children == 0:
        id_to_wnid[ilsvrc_id] = wnid

print("Classes:", len(id_to_wnid))
assert len(id_to_wnid) == 1000

with open(GT_PATH, "r") as f:
    labels = [int(line.strip()) for line in f if line.strip()]

print("Validation labels:", len(labels))
assert len(labels) == 50000

for index, label in enumerate(labels, start=1):
    wnid = id_to_wnid[label]
    filename = f"ILSVRC2012_val_{index:08d}.JPEG"
    src = os.path.join(VAL_DIR, filename)
    dst_dir = os.path.join(VAL_DIR, wnid)
    dst = os.path.join(dst_dir, filename)

    os.makedirs(dst_dir, exist_ok=True)

    if os.path.exists(src):
        shutil.move(src, dst)

    if index % 5000 == 0:
        print(f"Organized {index}/50000")

print("ImageNet validation organization completed.")
