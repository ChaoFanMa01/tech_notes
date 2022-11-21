import torch
from torch import nn


# Reference: 
#
# [1] A. Fischer, and C. Igel, "An Introduction to Restricted Boltzmann 
# Machines," LNCS, vol. 7441, pp. 14-36, 2012.
# [2] Y. Wang, Z. Pan, X. Yuan, C. Yang, and W. Gui, "A Novel Deep Learning 
# based Fault Diagnosis Approach for Chemical Process with Extended Deep 
# Belief Network," ISA Trans., vol. 96, pp.457-467, 2020.
#

class RBM(nn.Module):
    def __init__(self,
                 num_v,
                 num_h,
                 real_valued = True,
                 k = 1,
                 lr = 0.01,
                 num_epochs = 10,
                 batch_size = 20,
                 device = 'cpu'
                ):
        '''
        @param num_v: the number of visible variables.
        @param num_h: the number of hidden variables.
        @param real_valued: True if using real-valued RBM; 
                            False otherwise.
        @param k: k value used for CD-K algorithm.
        @param lr: learning rate.
        @param num_epochs: number of epochs.
        @param batch_size: size of batch train.
        @param device: the device to store parameters.
        '''
        super(RBM, self).__init__()
        self.num_v = num_v
        self.num_h = num_h
        self.real_valued = real_valued
        self.k = k
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        # All parameters are initialized to zero according to [1].
        # The weight matrix.
        self.w = torch.empty(self.num_h, self.num_v, dtype = torch.float32, 
                              device = self.device).uniform_(0., 0.01)
        self.w.requires_grad = True
        # The biases associated with visible variables.
        self.b = torch.zeros(self.num_v, 1, dtype = torch.float32, device = device,
                             requires_grad = True)
        # The biases associated with hidden variables.
        self.c = torch.zeros(self.num_h, 1, dtype = torch.float32, device = device,
                             requires_grad = True)
    
    def v2h_forward(self, x):
        '''
        The forward path from visible units to hidden units.
        NOTE: the shape of `x` should be (num_v, num_samples).
        '''
        if len(x.shape) < 2:
            v = x.reshape((-1, 1))
        elif len(x.shape) == 2:
            v = x
        else:
            raise RuntimeError('v2h_forward: shape mismatch for visible variable.')
        
        # Forward. 
        # NOTE: the broadcasting may involved.
        h = torch.mm(self.w, v) + self.c

        # For real-valued RBM.
        if self.real_valued:
            # Calculate the probability of $p(h|v)$.
            prob_h_v = h
            # Sample $h$ from the probability distribution computed previously.
            # In real-valued RBM, this is achieved simply by copy its probability.
            h = prob_h_v
        else:
            # Calculate the probability of $p(h|v)$.
            prob_h_v = torch.sigmoid(h)
            # Sample $h$ from the probability distribution computed previously.
            h = prob_h_v > torch.empty(prob_h_v.shape).uniform_(0., 1.)
            h = h.type(torch.float32)
        return prob_h_v, h
    
    def h2v_forward(self, x):
        '''
        The forward path from hidden units to visible units.
        NOTE: the shape of `x` should be (num_h, num_samples).
        '''
        if len(x.shape) < 2:
            h = x.reshape((-1, 1))
        elif len(x.shape) == 2:
            h = x
        else:
            raise RuntimeError('h2v_forward: shape mismatch for hidden variable.')
        
        # Forward. 
        # NOTE: the broadcasting may involved.
        v = torch.mm(h.T, self.w).T + self.b

        # For real-valued RBM.
        if self.real_valued:
            # Calculate the probability of $p(v|h)$.
            prob_v_h = v
            # Sample $v$ from the probability distribution computed previously.
            # In real-valued RBM, this is achieved simply by copy its probability.
            v = prob_v_h
        else:
            # Calculate the probability of $p(v|h)$.
            prob_v_h = torch.sigmoid(v)
            # Sample $v$ from the probability distribution computed previously.
            v = prob_v_h > torch.empty(prob_v_h.shape).uniform_(0., 1.)
            v = v.type(torch.float32)
        return prob_v_h, v
    
    def gibbs_sample(self, v):
        '''
        Gibbs sampling for CD-K algorithm.
        @param v: input visible variables whose shape is (num_v, num_samples).
        '''
        v0 = v
        prob_h_v0, h0 = self.v2h_forward(v0)
        prob_v_hk, vk = self.h2v_forward(h0)
        for k in range(self.k - 1):
            prob_h_vk, hk = self.v2h_forward(vk)
            prob_v_hk, vk = self.h2v_forward(hk)
        prob_h_vk, hk = self.v2h_forward(vk)
        return v0, vk, prob_h_v0, prob_h_vk
    
    def update_param(self, v0, vk, prob_h_v0, prob_h_vk):
        '''
        Update the parameters of RBM based on the Gibbs sample.
        @param v0: $v^{(0)}$. Shape: (num_v, num_samples).
        @param vk: $v^{(k)}$. Shape: (num_v, num_samples).
        @param prob_h_v0: $p(h_i=1|v^{(0)})$ for binary RBM, or
                          $h^{(0)}$ for real-valued RBM.
                          Shape: (num_h, num_samples).
        @param prob_h_vk: $p(h_i=1|v^{(k)})$ for binary RBM, or
                          $h^{(k)}$ for real-valued RBM.
                          Shape: (num_h, num_samples).
        '''
        num_samples = v0.shape[1]
        positives, negatives = [], []        
        for i in range(num_samples):
            positives.append(torch.mm(prob_h_v0[:, i].view((-1, 1)), v0[:, i].view(-1, 1).T).tolist())
            negatives.append(torch.mm(prob_h_vk[:, i].view((-1, 1)), vk[:, i].view(-1, 1).T).tolist())
        delta_w = torch.mean(torch.tensor(positives) - torch.tensor(negatives), axis = 0)
        delta_b = torch.mean(v0 - vk, axis = 1).view(-1, 1)
        delta_c = torch.mean(prob_h_v0 - prob_h_vk, axis = 1).view(-1, 1)
        
        delta_w = delta_w.to(self.device)
        delta_b = delta_b.to(self.device)
        delta_c = delta_c.to(self.device)

        self.w += self.lr * delta_w
        self.b += self.lr * delta_b
        self.c += self.lr * delta_c
    
    def cd_k(self, v):
        '''
        Tuning the parameters of the RBM based on the CD-K algorithm.
        @param v: training set, whose shape is (num_v, num_samples).
        '''
        num_samples = v.shape[1]
        batch_indices = num_samples // self.batch_size
        for i in range(batch_indices):
            print('CD-K: the %d-th batch' % (i))
            v0, vk, prob_h_v0, prob_h_vk = self.gibbs_sample(v[:, i * self.batch_size: i * self.batch_size + self.batch_size])
            self.update_param(v0, vk, prob_h_v0, prob_h_vk)
    
    def reconstruct(self, v):
        '''
        Reconstruct the given visible variables.
        '''
        _, h = self.v2h_forward(v)
        _, rec_v = self.h2v_forward(h)
        return rec_v
    
    def fit(self, v):
        '''
        The train process of RBM.
        @param v: training set whose shape is (num_v, num_samples).
        '''
        with torch.no_grad():
            for i in range(self.num_epochs):
#                print('The %d-th CD-K' % (i))
                self.cd_k(v)
                rec_v = self.reconstruct(v)
                l = self.loss(v, rec_v)
#                print('The loss of the %d-th epoch is %.2f' % (i, l))
    
    def loss(self, v, rec_v):
        '''
        The loss of the RBM. The reconstruction error is used currently.
        @param v: the visible variables for test. Shape: (num_v, num_samples).
        @param rec_v: the reconstruction of `v` by the RBM.
        '''
        l = torch.pow(torch.sum(torch.pow(v - rec_v, 2), axis = 0), 0.5)
        return torch.mean(l)