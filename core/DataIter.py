#!/usr/bin/python


import mxnet as mx
import numpy as np
import config
import core.visualize




def get_mnist_iter():
    trans = lambda data, label: (mx.nd.transpose(data.astype(np.float32)/128-1, axes=(2,0,1)), label.astype(np.uint8))

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



if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath(os.curdir))
    sys.path.append(os.path.abspath(os.curdir)+'/..')
    sys.path.append(os.path.abspath(os.curdir)+'/../core')

    nmist_iter = get_mnist_iter()

    for batch in nmist_iter:
        b0 = batch[0]

        img = np.ceil((b0.asnumpy()+1)*128).astype(np.uint8)
        print(img)
        visulize.show_image(img)


        break

