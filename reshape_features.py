import os
import numpy as np

feat_dir = './data/sics/features'

files = os.listdir(feat_dir)

for f in files:
    if f.endswith('.npy'):
        feat = np.load(os.path.join(feat_dir, f))
        feat = feat.transpose(1, 0)
        np.save(os.path.join(feat_dir, f), feat)