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

class Logger:
    def __init__(self):
        self.buffer = []
        self.enabled = False

    def init(self):
        """Start / reset logging."""
        self.buffer = []
        self.enabled = True

    def __call__(self, *args, sep=" ", end="\n", file=None, flush=False, say=True):
        """
        Drop-in replacement for print:
        - prints to stdout
        - also appends to internal buffer if enabled
        """
        msg = sep.join(str(a) for a in args)
        # print normally
        if say:
            print(msg, sep=sep, end=end, file=file, flush=flush)
        # and store
        if self.enabled:
            self.buffer.append(msg + ("" if end == "" else end))

    def saveto(self, path: str):
        """Save the collected log to a file at `path`."""
        text = "".join(self.buffer)
        # use your existing helper
        writef(path, text)



