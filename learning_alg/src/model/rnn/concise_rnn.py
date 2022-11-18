import torch
from torch import nn

class RNNModel(nn.Module):
  def __init__(self, rnn_layer, vocab_size):
    super(RNNModel, self).__init__()
    self.rnn = rnn_layer
    self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bindirectional  else 1)
    self.vocab_size = vocab_size
    self.dense = nn.Linear(self.hidden_size, vocab_size)
    self.state = None

  def forward(self, inputs, state):
    X = misc.to_onehot(inputs, self.vocab_size)
    Y, self.state = self.rnn(torch.stack(X), state)
