import os
import random
import logging
import numpy as np
import torch

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/execution.log", mode='w', encoding="utf-8"),
        logging.StreamHandler()
    ]
)

RANDOM_STATE = 42
DATA_FILE = "data/indianpinearray.npy"
GT_FILE = "data/IPgt.npy"

TARGET_ACCURACY = 0.93
TARGET_F1_MACRO = 0.90

def seed_everything(seed=RANDOM_STATE):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")