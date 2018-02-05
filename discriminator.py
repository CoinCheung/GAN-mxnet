#!/usr/bin/python


import mxnet as mx
import numpy as np
import config
import IO


def discriminator_conv(img):

    conv1 = mx.sym.Convolution(img, num_filter=32, kernel=(5,5), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv1')
    leaky1 = mx.sym.LeakyReLU(conv1, act_type='leaky', slope=0.01, name='dis_leaky_relu1')
    maxpool1 = mx.sym.Pooling(leaky1, kernel=(2,2), pool_type='max', stride=[2,2], name='dis_max_pooling1')

    conv2 = mx.sym.Convolution(maxpool1, num_filter=64, kernel=(5,5), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv2')
    leaky2 = mx.sym.LeakyReLU(conv2, act_type='leaky', slope=0.01, name='dis_leaky_relu2')
    maxpool2 = mx.sym.Pooling(leaky2, kernel=(2,2), pool_type='max', stride=[2,2], name='dis_max_pooling2')

    fc1 = mx.sym.FullyConnected(maxpool2, num_hidden=4*4*64, no_bias=False, flatten=True, name='dis_fc1')
    leaky3 = mx.sym.LeakyReLU(fc1, act_type='leaky', slope=0.01, name='dis_leaky_relu3')

    fc2 = mx.sym.FullyConnected(leaky3, num_hidden=1, no_bias=False, flatten=True, name='dis_fc2')

    out = fc2

    return out


if __name__ == "__main__":
    batch_size = config.batch_size
    img_size = config.img_size

    img = mx.sym.var('img')
    dis = discriminator_conv(img)
    mod = mx.mod.Module(dis, context=mx.cpu(), data_names=['img'], label_names=None)
    mod.bind(data_shapes=[('img',(batch_size,1,img_size,img_size))], label_shapes=None)
    mod.init_params()

    it = IO.get_mnist_iter()
    #  batch = it.next()[0]
    for img, label in it:
        print(img.shape)
        print(label.shape)

        batch = mx.io.DataBatch([img], label=None)

        mod.forward(batch)

        out = mod.get_outputs()
        print(out[0])

        break



