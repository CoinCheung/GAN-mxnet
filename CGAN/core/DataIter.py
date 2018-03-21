#!/usr/bin/python


import mxnet as mx
import numpy as np
import core.config as config


batch_size = config.batch_size


def trans(data, label):
    data = mx.nd.transpose(data.astype(np.float32)/127.5-1, axes=(2,0,1))
    label.astype(np.float32)
    return data, label

def get_mnist_iter():
    mnist_train = mx.gluon.data.vision.MNIST(root='~/.mxnet/datasets/mnist/',
                                             train=True, transform=trans)
    train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)

    return train_data
