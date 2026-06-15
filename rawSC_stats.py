import numpy as np
import matplotlib.pyplot as plt
import os

def remap_labels(lbl):
    the_lbl = np.unique(lbl)
    remap = { x: i for i,x in enumerate(the_lbl)}
    new_lbl = np.vectorize(remap.get)(lbl)
    return new_lbl

basename = os.path.expanduser("~/data/rawSC/")
plt.figure()
for split in ["train", "validation", "test"]:
    l = np.load(basename+"labels_"+split+".npy")
    plt.hist(l,bins=np.arange(0,35))

plt.figure()
for split in ["train", "validation", "test"]:
    u = np.load(basename+"speaker_"+split+".npy")
    plt.hist(u,bins=np.arange(0,2607))
plt.show()
