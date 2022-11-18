import torch
from torch import nn
import numpy as np
import random
import math
import time

def load_data_jay_lyrics():
  with open('../../data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[: 10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]

    return corpus_indices, char_to_idx, idx_to_char, vocab_size

def data_iter_random(corpus_indices, batch_size, num_steps, device = None):
  num_examples = (len(corpus_indices) - 1) // num_steps
  epoch_size = num_examples // batch_size 
  example_indices = list(range(num_examples))
  random.shuffle(example_indices)

  def _data(pos):
    return corpus_indices[pos: pos + num_steps]

  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  for i in range(epoch_size):
    i *= batch_size
    batch_indices = example_indices[i: i + batch_size]
    X = [_data(j * num_steps) for j in batch_indices]
    Y = [_data(j * num_steps + 1) for j in batch_indices]
    yield torch.tensor(X, dtype = torch.float32, device = device),\
          torch.tensor(Y, dtype = torch.float32, device = device)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, device = None):
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  corpus_indices = torch.tensors(corpus_indices, dtype = torch.float32, device = device)
  data_len = len(corpus_indices)
  batch_len = data_len // batch_size
  indices = corpus_indices[: batch_size * batch_len].view(batch_size, batch_len)
  epoch_size = (batch_len - 1) // num_steps
  for i in range(epoch_size):
    i = i * num_steps
    X = indices[:, i: i + num_steps]
    Y = indices[:, i + 1: i + num_steps + 1]
    yield X, Y

def one_hot(x, n_class, dtype = torch.float32):
  # X shape: (batch), output shape: (batch, n_class)
  x = x.long()
  res = torch.zeros(x.shape[0], n_class, dtype = dtype, device = x.device)
  res.scatter_(1, x.view(-1, 1), 1)
  return res

def to_onehot(X, n_class):
  return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

def get_params(num_inputs, num_hiddens, num_outputs, device):
  def _one(shape):
    ts = torch.tensor(np.random.normal(0, 0.01, size = shape), device = device, dtype = torch.float32)
    return torch.nn.Parameter(ts, requires_grad = True)

  W_xh = _one((num_inputs, num_hiddens))
  W_hh = _one((num_hiddens, num_hiddens))
  b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device = device, requires_grad = True))

  W_hq = _one((num_hiddens, num_outputs))
  b_q = torch.nn.Parameter(torch.zeros(num_outputs, device = device, requires_grad = True))
  return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

# 初始化隐藏状态.
def init_rnn_state(batch_size, num_hiddens, device):
  return (torch.zeros((batch_size, num_hiddens), device = device), )

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char,
                char_to_idx):
  state = init_rnn_state(1, num_hiddens, device)
  output = [char_to_idx[prefix[0]]]
  for t in range(num_chars + len(prefix) - 1):
    X = to_onehot(torch.tensor([[output[-1]]], device = device), vocab_size)
    (Y, state) = rnn(X, state, params)
    if t < len(prefix) - 1:
      output.append(char_to_idx[prefix[t + 1]])
    else:
      output.append(int(Y[0].argmax(dim = 1).item()))
  return ''.join([idx_to_char[i] for i in output])

def grad_clipping(params, theta, device):
  norm = torch.tensor([0.], device = device)
  for param in params:
    norm += (param.grad.data ** 2).sum()
  norm = norm.sqrt().item()
  if norm > theta:
    for param in params:
      param.grad.data *= (theta / norm)

def sgd(params, lr, batch_size):
  for param in params:
    param.data -= lr * param.grad / batch_size

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
  if is_random_iter:
    data_iter_fn = data_iter_random
  else:
    data_iter_fn = data_iter_consecutive
  params = get_params(vocab_size, num_hiddens, vocab_size, device)
  loss = nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
    if not is_random_iter:
      state = init_rnn_state(batch_size, num_hiddens, device)
    l_sum, n, start = 0., 0, time.time()
    data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
    for X, Y in data_iter:
      if is_random_iter:
        state = init_rnn_state(batch_size, num_hiddens, device)
      else:
        for s in state:
          s.detach_()
      
      inputs = to_onehot(X, vocab_size)
      (outputs, state) = rnn(inputs, state, params)
      outputs = torch.cat(outputs, dim = 0)
      y = torch.transpose(Y, 0, 1).contiguous().view(-1)
      l = loss(outputs, y.long())

      if params[0].grad is not None:
        for param in params:
          param.grad.data.zero_()
      l.backward()
      grad_clipping(params, clipping_theta, device)
      sgd(params, lr, 1)
      l_sum += l.item() * y.shape[0]
      n += y.shape[0]

    if (epoch + 1) % pred_period == 0:
      print('epoch % d, perplexity %f, time %.2f sec' % 
            (epoch + 1, math.exp(l_sum / n), time.time() - start))
      for prefix in prefixes:
        print(' -', predict_rnn(prefix, pred_len, rnn,
                     params, init_rnn_state,num_hiddens, vocab_size, 
                    device, idx_to_char,char_to_idx))
