import torch

def rnn(inputs, state, params):
  W_xh, W_hh, b_h, W_hq, b_q = params
  H, = state
  outputs = []
  for X in inputs:
    H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
    Y = torch.matmul(H, W_hq) + b_q
    outputs.append(Y)
  return outputs, (H,)
