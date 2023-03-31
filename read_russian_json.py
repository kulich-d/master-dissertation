import json
import os

import numpy as np

save_path = ""  # path to russian json


def read_json(save_path):
    with open(os.path.join(save_path, "report_information.json"), encoding='ascii') as f:
        data = json.load(f)
    print(data)
    for k, v in data.items():
        v = [float(i) for i in v]
        print(f"{k}: median {round(np.median(v), 2)}, max {round(np.max(v), 2)},  min {round(np.min(v), 2)},  ")
        print(v)
