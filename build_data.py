import sys
import dataset
from vibgen import VibConfig
dataset.get("train", 3000, 1)
dataset.get("test_nominal", 1200, 2, ideal=True)
dataset.get("test_highnoise", 600, 3, VibConfig(noise_scale=2.5))
dataset.get("test_heavydrop", 600, 4, VibConfig(p_drop=0.15, p_drift=0.10))
dataset.get("test_weakfault", 600, 5, VibConfig(sev_range=(0.1, 0.4)))
print("done")
