import os
import torch

# data & model
DATA_PATH    = "cmems_obs-sst_glo_phy_nrt_l4_P1D-m_1749202248617.nc"
MODEL_DIR    = "./checkpoints"
LOG_DIR      = "./logs"

# training
BATCH_SIZE   = 16
LR           = 1e-3
EPOCHS       = 15
PATIENCE     = 10
WEIGHT_DECAY = 0.0

# sequence lengths
T_IN, T_OUT  = 10, 7

# physical loss weights
λ_G, λ_V     = 0.1, 0.05

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"----{DEVICE}------")

# make dirs
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
