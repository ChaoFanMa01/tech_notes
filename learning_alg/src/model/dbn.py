import torch
from torch import nn
from torch import optim
import sys

from model.rbm_torch import RBM

# Reference: 
#
# [1] Y. Bengio, P. Lamblin, D. Popovici, and H. Larochelle, "Greedy Layer-Wise
#     Training of Deep Networks," in proc. NIPS'06, 2006.

class DBN(nn.Module):
    '''
    Pytorch implementation of the Deep Belief Network (DBN).
    '''
    def __init__(self,
                 num_vs,
                 num_hs,
                 num_out,
                 classifier = True,
                 real_valued = True,
                 k = 1,
                 lr = 0.01,
                 pretrain_epochs = 10,
                 finetune_epochs = 100,
                 batch_size = 20,
                 device = 'cpu'
                ):
        '''
        @param num_vs: list specifies the number of visible variables at each layer.
        @param num_hs: list specifies the number of hidden variables at each layer.
        @param num_out: the number of units at the output layer.
        @param classifier: True if the model is to be used as a classifier;
                           False if the model is to be used as a regressor.
        @param real_valued: True if using real-valued RBM; 
                            False otherwise.
        @param k: k value used for CD-K algorithm.
        @param lr: learning rate.
        @param pretrain_epochs: number of epochs for pre-train process.
        @param finetune_epochs: number of epochs for fine-tuning process.
        @param batch_size: size of batch train.
        @param device: the device to store parameters.
        '''
        super(DBN, self).__init__()
        if len(num_vs) != len(num_hs):
            raise RuntimeError('DBM initializer: parameter error.')
        self.num_vs = num_vs
        self.num_hs = num_hs
        self.num_out = num_out
        self.classifier = classifier
        self.real_valued = real_valued
        self.k = k
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.device = device
        self.num_layers = len(num_vs)

        self.rbms = [RBM(num_vs[i], num_hs[i], real_valued, k, 
                        lr, pretrain_epochs, batch_size, device) for 
                        i in range(self.num_layers)]
        self.dense = nn.Linear(num_hs[-1], num_out, device = device)
    
    def forward(self, x):
        '''
        The forward path of the DBM.
        NOTE: the shape of `x` should be (num_v, num_samples).
        '''
        v = x
        for i in range(self.num_layers):
            prob_h_v, h = self.rbms[i].v2h_forward(v)
            v = h
        out = self.dense(v.T)
        return out
    
    def reconstruct(self, x):
        '''
        Reconstruct the given visible variables.
        '''
        v = x
        for i in range(self.num_layers):
            _, h = self.rbms[i].v2h_forward(v)
            v = h
        for i in range(self.num_layers):
            _, v = self.rbms[self.num_layers - i - 1].h2v_forward(h)
            h = v
        return v
    
    def pre_train(self, v):
        '''
        The pre-train process of DBM. The greedy layer-wise training 
        scheme is used in this process.
        '''
        with torch.no_grad():
            for i in range(self.num_layers):
                print('pre-train: %d-th layer' % (i))
                self.rbms[i].fit(v)
                _, v = self.rbms[i].v2h_forward(v)
    
    def fit(self, v, y):
        '''
        Training of the DBM. This process includes the pre-train process
        and the fine-tuning process.
        @param v: the training set.
        @param y: the labels of training set.
        '''
        # pre-train process.
        self.pre_train(v)

        # fine-tuning process.
        loss_fn = nn.CrossEntropyLoss() if self.classifier else nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr = self.lr)
        for i in range(self.finetune_epochs):
            print('fine tuning: %d-th epoch' % (i))
            hat_y = self.forward(v)
            print(hat_y)
            print(y)
            l = loss_fn(hat_y, torch.squeeze(y.view(1, -1).type(torch.long)))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()