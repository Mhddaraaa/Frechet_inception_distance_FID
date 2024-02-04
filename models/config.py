import torch

EPOCH = 20
Z_DIM = 100
LR = 2e-4
BS = 64
C, H, W = 1, 32, 32
NUM_CLASS = 10 # 0, 1, 2, ..., 9
EMBED_SIZE = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'