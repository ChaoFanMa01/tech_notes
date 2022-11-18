#! /usr/bin/env python3

import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import misc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = misc.load_data_jay_lyrics()
num_hiddens = 256

