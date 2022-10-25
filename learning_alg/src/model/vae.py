import torch
import torch.nn as nn
import torch.nn.functional as F

features = 16

class LinearVAE(nn.Module):
  def __init__(self):
    super(LinearVAE, self).__init__()

    # encoder
    self.enc1 = nn.Linear(in_features = 784, out_features = 512)
    self.enc2 = nn.Linear(in_features = 512, out_features = features * 2)

    # decoder
    self.dec1 = nn.Linear(in_features = features, out_features = 512)
    self.dec2 = nn.Linear(in_features = 512, out_features = 784)

  def reparameterize(self, mu, log_var):
    '''
    @param mu: mean from the encoder's latent space.
    @param log_var: log variance from the encoder's latent space.
    '''
    # 文章里说`EncoderNeuralNet`出来的直接是$\log\sigma$.
    # 但是如果按照这个计算来看的话, `EncoderNeuralNet`出
    # 来的是$\log\sigma^2$.而且, 好多实现都是这样实现的.
    # 另外, 感觉差不差这个0.5, 对结果影响不大, 需要后续验
    # 证.
    std = torch.exp(0.5 * log_var)
    # 这里面没有用高斯分布进行采样, 存在问题.
    eps = torch.randn_like(std)
    # 生成对latent variable `z`的采样.
    sample = mu + (eps * std)
    return sample

  def forward(self, x):
    # encoding
    x = F.relu(self.enc1(x))
    x = self.enc2(x).view(-1, 2, features)

    # get `mu` and `log_var`
    mu = x[:, 0, :]
    log_var = x[:, 1, :]

    # reparameterization
    z = self.reparameterize(mu, log_var)

    # decoding
    x = F.relu(self.dec1(z))
    reconstruction = torch.sigmoid(self.dec2(x))
    return reconstruction, mu, log_var
