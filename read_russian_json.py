import json
import os

import numpy as np

save_path = "/Users/diana.kulich/Documents/Masters/dissertation/exp/Karjan_girl_all_processing_new_anglesch01_20210909192521"

with open(os.path.join(save_path, "all_information.json"), encoding='ascii') as f:
    data = json.load(f)
print(data)
for k, v in data.items():
    v = [float(i) for i in v]
    print(f"{k}: median {round(np.median(v), 2)}, max {round(np.max(v), 2)},  min {round(np.min(v), 2)},  ")
    print( v)
