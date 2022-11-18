#! /usr/bin/env python3

import numpy as np
import torch

import misc
from scratch_rnn import rnn

corpus_indices, char_to_idx, idx_to_char, vocab_size = misc.load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs, lr, clipping_theta, num_steps, batch_size = 250, 1e2, 1e-2, 25, 32
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
misc.train_and_predict_rnn(rnn, misc.get_params, misc.init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices,idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period,
                      pred_len,prefixes)
