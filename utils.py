import os
import logging
import random
import yaml
from pathlib import Path
from os import PathLike
from typing import Any, List, Dict

import numpy as np
import torch

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)

def load_yaml(path: PathLike) -> Any:
    with open(path) as f:
        obj = yaml.safe_load(f)
    return obj

def dump_yaml(obj: Any, path: PathLike) -> None:
    with open(path, 'w') as f:
        yaml.dump(obj, f)

class AverageMeter:
    def __init__(self, *keys):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}
        self.avgs = {key: 0.0 for key in keys}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1
            self.avgs[key] = self.totals[key] / self.counts[key]

    def _check_attr(self, attr):
        assert attr in self.totals and attr in self.counts, f"{attr} not found in AverageMeter"

    def __getattr__(self, attr):
        self._check_attr(attr)
        return self.avgs[attr]

    def __repr__(self):
        return ', '.join(f'{key}: {value:.4f}' for key, value in self.avgs.items())
