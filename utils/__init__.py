# utils.py
import os, re, argparse
import json, pickle

import torch
import random
import numpy as np

import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm 

# % --- load and save functions ---
def readf(path):
    with open(path, 'r') as f:
        return f.read()

def writef(path, content, encoding="utf-8"):
    with open(path, 'w') as f:
        f.write(content)

class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(filepath, dictionary):
    with open(filepath, "w") as f:
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

def loadjl(filepath):
    filepath = Path(filepath)
    items = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def dumpp(filepath, obj):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def loadp(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

        
# % --- logging & seed ---

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

