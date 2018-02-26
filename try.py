#!/usr/bin/python


from cffi.cwrapper import ffi, lib

lib.resize_wrap()


import core.DataIter as DI
import mxnet as mx
import numpy as np


def trans(data, label):
    return mx.nd.transpose(data.astype(np.float32)/128-1, axes=(0,1,2)), label.astype(np.uint8)

mnist_train = mx.gluon.data.vision.MNIST(root='~/.mxnet/datasets/mnist/', train=True, transform=trans)
it = mx.gluon.data.DataLoader(mnist_train, 64, shuffle=True)

generator =  it.__iter__()
data = generator.send(None)[0][0]

#  data = mx.nd.transpose(data, (1,2,0))

#  print(data.shape)

dd = mx.image.resize_short(data, 64)
#  dd = mx.image.imresize(data,64,64)

#  print(dd.shape)


########
dat = mx.sym.var('data')
out = mx.sym.sum(dat)

mod = mx.mod.Module(out, context=mx.cpu(), data_names=['data'], label_names=None)
mod.bind(data_shapes=[('data',(1, 3))], label_shapes=None)
mod.init_params(initializer=mx.initializer.Xavier())

data = mx.nd.ones((10, 3),ctx=mx.cpu())
batch = mx.io.DataBatch([data],label=None)
mod.forward(batch)

out_val = mod.get_outputs()[0].asnumpy()
print(out_val)


class Myclass(object):
    def __init__(self):
        self._name = 'coin'


mc = Myclass()
print(mc._name)
