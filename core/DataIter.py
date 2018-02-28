#!/usr/bin/python


import mxnet as mx
import numpy as np
import core.config as config


def trans(data, label):
    data = mx.img.imresize(data,64,64)
    data = mx.nd.transpose(data.astype(np.float32)/127.5-1, axes=(2,0,1))
    label.astype(np.uint8)
    return data, label


def get_mnist_iter():
    mnist_train = mx.gluon.data.vision.MNIST(root='~/.mxnet/datasets/mnist/', train=True, transform=trans)
    train_data = mx.gluon.data.DataLoader(mnist_train, config.batch_size, shuffle=True)

    return train_data



def gen_noise_uniform(shape, bound):
    '''
    generate a unified noise symbol with given shape
    params:
        shape: a tuple of the noise matrix shape of (batch_size, noise_dim)
        bound: list or tuple, the bound of the noises
    return:
        a nd array
    '''
    return mx.nd.random.uniform(bound[0], bound[1], shape=shape)

