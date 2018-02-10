#!/usr/bin/python


import mxnet as mx
import numpy as np
import config
import symbol.loss as loss
import core.IO as IO


def discriminator_conv(img, label, batch_size, test, leaky_slope, eps):

### old net
    conv1 = mx.sym.Convolution(img, num_filter=32, kernel=(5,5), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv1')
    #  bn1 = mx.sym.BatchNorm(conv1, fix_gamma=False, use_global_stats=test, eps=eps, name='dis_bn1')
    leaky1 = mx.sym.LeakyReLU(conv1, act_type='leaky', slope=0.01, name='dis_leaky_relu1')
    maxpool1 = mx.sym.Pooling(leaky1, kernel=(2,2), pool_type='max', stride=[2,2], name='dis_max_pooling1')

    conv2 = mx.sym.Convolution(maxpool1, num_filter=64, kernel=(5,5), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv2')
    #  bn2 = mx.sym.BatchNorm(conv2, fix_gamma=False, use_global_stats=test, eps=eps, name='dis_bn2')
    leaky2 = mx.sym.LeakyReLU(conv2, act_type='leaky', slope=0.01, name='dis_leaky_relu2')
    maxpool2 = mx.sym.Pooling(leaky2, kernel=(2,2), pool_type='max', stride=[2,2], name='dis_max_pooling2')

    reshaped = mx.sym.reshape(maxpool2, shape=(batch_size, -1))

    fc1 = mx.sym.FullyConnected(maxpool2, num_hidden=4*4*64, no_bias=False, flatten=True, name='dis_fc1')
    #  bn3 = mx.sym.BatchNorm(fc1, fix_gamma=False, use_global_stats=test, eps=eps, name='dis_bn3')
    leaky3 = mx.sym.LeakyReLU(fc1, act_type='leaky', slope=0.01, name='dis_leaky_relu3')

    fc2 = mx.sym.FullyConnected(leaky3, num_hidden=1, no_bias=False, flatten=True, name='dis_fc2')

    CE, logits_sigmoid = loss.sigmoid_cross_entropy(fc2, label, batch_size)

    CE_loss = mx.sym.MakeLoss(CE)
    out = mx.sym.Group([CE_loss, mx.sym.BlockGrad(logits_sigmoid)])
### new net

    #  # 28x28x1
    #  tanh1 = mx.sym.Activation(img, act_type='tanh', name='dis_tanh1')
    #  conv1 = mx.sym.Convolution(tanh1, num_filter=64, kernel=(4,4), stride=(2,2), pad=(1,1), no_bias=True, name='dis_conv1')
    #  leaky1 = mx.sym.LeakyReLU(conv1, act_type='leaky', slope=leaky_slope, name='dis_leaky_relu1')
    #  bn1 = mx.sym.BatchNorm(leaky1, fix_gamma=False, use_global_stats=test, eps=eps, name='dis_bn1')
    #  # 14x14x64
    #  conv2 = mx.sym.Convolution(bn1, num_filter=128, kernel=(4,4), stride=(2,2), pad=(1,1), no_bias=True, name='dis_conv2')
    #  leaky2 = mx.sym.LeakyReLU(conv2, act_type='leaky', slope=leaky_slope, name='dis_leaky_relu2')
    #  bn2 = mx.sym.BatchNorm(leaky2, fix_gamma=False, use_global_stats=test, eps=eps, name='dis_bn2')
    #  # 7x7x128
    #  conv3 = mx.sym.Convolution(bn2, num_filter=256, kernel=(3,3), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv3')
    #  leaky3 = mx.sym.LeakyReLU(conv3, act_type='leaky', slope=leaky_slope, name='dis_leaky_relu3')
    #  bn3 = mx.sym.BatchNorm(leaky3, fix_gamma=False, use_global_stats=test, eps=eps, name='dis_bn3')
    #  # 5x5x256
    #  conv4 = mx.sym.Convolution(bn3, num_filter=256, kernel=(3,3), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv4')
    #  leaky4 = mx.sym.LeakyReLU(conv4, act_type='leaky', slope=leaky_slope, name='dis_leaky_relu4')
    #  bn4 = mx.sym.BatchNorm(leaky4, fix_gamma=False, use_global_stats=test, eps=eps, name='dis_bn4')
    #  # 3x3x256
    #  conv5 = mx.sym.Convolution(bn4, num_filter=1, kernel=(3,3), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv5')
    #  # 1x1x1
    #
    #  logits = conv5.reshape(-1,)
    #
    #  CE, logits_sigmoid = loss.sigmoid_cross_entropy(logits, label, batch_size)
    #
    #  CE_loss = mx.sym.MakeLoss(CE)
    #  out = mx.sym.Group([CE_loss, mx.sym.BlockGrad(logits_sigmoid)])

    return out


if __name__ == "__main__":
    batch_size = config.batch_size
    img_size = config.img_size

    img = mx.sym.var('img')
    dis = discriminator_conv(img, batch_size)
    mod = mx.mod.Module(dis, context=mx.cpu(), data_names=['img'], label_names=None)
    mod.bind(data_shapes=[('img',(batch_size,1,img_size,img_size))], label_shapes=None)
    mod.init_params(mx.initializer.Xavier())

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



