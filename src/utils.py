import os, random, json, torch, numpy as np

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device(arg:str="auto"):
    import torch
    return torch.device("cuda" if arg=="auto" and torch.cuda.is_available() else "cpu")

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)
