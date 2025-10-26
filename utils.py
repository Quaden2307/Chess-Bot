import torch
import pandas
import numpy as np
import chess
import matplotlib as plt    
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

PIECE_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_tensor(board: chess.Board):

    tensor = np.zeros((8,8,12), dtype = np.float32)

    for square, piece in board.piece_map():
        x = chess.square_file(square)
        y = chess.square_rank(square)
        tensor[y, x, PIECE_INDEX[piece.symbol()]] = 1.0

    return torch.from_numpy(tensor).flatten()
