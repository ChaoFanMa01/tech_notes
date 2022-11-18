#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from model.rbm import RBM
from model.mnist_loader import load_train_images

def test():
    data = load_train_images()
    data /= 255.
    h, w = data[0].shape
    num_v = h * w
    num_h = int(0.5 * num_v)
    real_valued = True
    k = 1
    lr = 0.001
    num_epochs = 1
    batch_size = 1
    v = np.copy(data)
    # Transform the shape of data to the form of (num_v, num_samples).
    v = np.squeeze(v.reshape(-1, num_v, 1).T)

    rbm = RBM(num_v, num_h, real_valued, k, lr, num_epochs, batch_size)
    rbm.fit(v)
    print('w ', rbm.w)
    print('b ', rbm.b.reshape((1, -1)))
    print('c ', rbm.c.reshape((1, -1)))
    prev = None
    for i in range(20):
        fig, axies = plt.subplots(nrows = 1, ncols = 2)
        rec_v = rbm.reconstruct(v[:, i].reshape(-1, 1))
        rec_v = rec_v.reshape((h, w))
        if prev is not None:
            print('diff: ', np.mean(rec_v - prev))
        prev = rec_v
        axies[0].imshow(rec_v, 'gray')
        axies[1].imshow(v[:, i].reshape(h, w), 'gray')
        plt.show()

if __name__ == '__main__':
    test()