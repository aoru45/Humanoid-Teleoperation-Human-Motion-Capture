import glob
import joblib
from easydict import EasyDict
from tqdm import tqdm
import os

data_ref = joblib.load("./resources/motions/h1/amass_phc_filtered.pkl")
keys = list(data_ref.keys())
keys_new = []
for k in keys:
    pre = k.split("-")[0]
    post = k.split("_")[1:]
    k_new = pre + "-" + "_".join(post)
    keys_new.append(k_new)
# 0-KIT_572_dance_chacha04_poses
# 0-572_dance_chacha12_poses.pkl
# 0-rub085_0014_knocking2_poses
# print(keys)


d = {}
files = glob.glob("./data/g1/motions/*.pkl")
for _file in tqdm(files):
    data = joblib.load(_file)
    name = os.path.basename(_file)[:-4]
    if not(name in keys_new):
        #print("continue", name)
        continue
    if len(data["pose_aa"].shape) != 3:
        print(_file)
        continue
    else:
        d[name] = data
        print("set", name)
    # for k,v in data.items():
    #     if hasattr(v, "shape"):
    #         print(k, v.shape)
    # exit()
joblib.dump(d, "./data/g1/amass_filtered.pkl")

