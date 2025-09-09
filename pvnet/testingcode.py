import os
print(os.path.abspath("../datasets/LINEMOD/cat/pose"))
import glob
files = glob.glob("../datasets/LINEMOD/cat/pose/*.npy")
print(len(files), files[:5])
import numpy as np
import glob, os

npy_files = glob.glob("../datasets/LINEMOD/cat/pose/*.npy")
for f in npy_files:
    arr = np.load(f)
    txt_name = f.replace(".npy", ".txt")
    np.savetxt(txt_name, arr)
