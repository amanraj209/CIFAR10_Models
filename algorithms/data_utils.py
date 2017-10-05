# Write function to load the cifar-10 data
# The original code is from http://cs231n.github.io/assignment1/
# The function is in data_utils.py file for reusing.
import numpy as np
import pandas as pd
import os

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    datadict = pd.read_pickle(filename)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
    Y = np.array(Y)
    return X, Y
        

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
