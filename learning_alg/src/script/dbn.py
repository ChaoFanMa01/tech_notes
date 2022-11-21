#! /usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from model.dbn import DBN
from model.mnist_loader import load_train_images, load_train_labels, \
                               load_test_images, load_test_labels

def test():
    train_data = load_train_images()
    test_data = load_test_images()
    train_labels = load_train_labels()
    test_labels = load_test_labels()
    train_data /= 255.
    h, w = train_data[0].shape
    num_v = h * w
    num_vs = [num_v, int(num_v / 2), int(num_v / 4), int(num_v / 8)]
    num_hs = [int(num_v / 2), int(num_v / 4), int(num_v / 8), int(num_v / 16)]
    num_out = 10
    classifier = True
    real_valued = True
    k = 1
    pretrain_epochs = 1
    finetune_epochs = 100
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = torch.tensor(0.001, device = device)
    train_v = np.copy(train_data)
    # Transform the shape of data to the form of (num_v, num_samples).
    train_v = np.squeeze(train_v.reshape(-1, num_v, 1).T)
    train_v = torch.tensor(train_v, dtype = torch.float32, device = device)
    train_y = torch.tensor(train_labels, device = device)

    test_v = np.copy(test_data)
    test_v = np.squeeze(test_v.reshape(-1, num_v, 1).T)
    test_v = torch.tensor(test_v, dtype = torch.float32, device = device)
    test_y = torch.tensor(test_labels, device = device)

    dbn = DBN(num_vs, num_hs, num_out, classifier, real_valued, k, lr, 
              pretrain_epochs, finetune_epochs, batch_size, device)
    dbn.fit(train_v, train_y)
    for i in range(20):
        with torch.no_grad():
            fig, axies = plt.subplots(nrows = 1, ncols = 2)
            rec_v = dbn.reconstruct(test_v[:, i].view(-1, 1))
            rec_v = rec_v.cpu()
            pred = dbn(test_v[:, i].view(-1, 1))
            pred = pred.cpu()
            rec_v = rec_v.view((h, w))
            axies[0].imshow(rec_v, 'gray')
            s = test_v[:, i].cpu()
            axies[1].imshow(s.reshape(h, w), 'gray')
            plt.show()
            print('pred: %d, label: %d' % (torch.argmax(pred), test_y[i].cpu()))

if __name__ == '__main__':
    test()