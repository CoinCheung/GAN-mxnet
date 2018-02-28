#!/usr/bin/python


import mxnet as mx
import symbol.loss as loss


def discriminator_lenet5(img, label, batch_size, leaky_slope):

    # ncx64x64
    conv1 = mx.sym.Convolution(img, num_filter=32, kernel=(5,5), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv1')
    leaky1 = mx.sym.LeakyReLU(conv1, act_type='leaky', slope=leaky_slope, name='dis_leaky_relu1')
    maxpool1 = mx.sym.Pooling(leaky1, kernel=(2,2), pool_type='max', stride=[2,2], name='dis_max_pooling1')
    # 32x32x32
    conv2 = mx.sym.Convolution(maxpool1, num_filter=64, kernel=(5,5), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv2')
    leaky2 = mx.sym.LeakyReLU(conv2, act_type='leaky', slope=leaky_slope, name='dis_leaky_relu2')
    maxpool2 = mx.sym.Pooling(leaky2, kernel=(2,2), pool_type='max', stride=[2,2], name='dis_max_pooling2')
    # 16x16x64
    conv3 = mx.sym.Convolution(maxpool2, num_filter=128, kernel=(5,5), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv3')
    leaky3 = mx.sym.LeakyReLU(conv3, act_type='leaky', slope=leaky_slope, name='dis_leaky_relu3')
    maxpool3 = mx.sym.Pooling(leaky3, kernel=(2,2), pool_type='max', stride=[2,2], name='dis_max_pooling3')
    # 8x8x128
    flatten = mx.sym.Flatten(maxpool3)

    fc1 = mx.sym.FullyConnected(flatten, num_hidden=4*4*64, no_bias=False, flatten=True, name='dis_fc1')
    leaky3 = mx.sym.LeakyReLU(fc1, act_type='leaky', slope=leaky_slope, name='dis_leaky_relu3')

    fc2 = mx.sym.FullyConnected(leaky3, num_hidden=1, no_bias=False, flatten=True, name='dis_fc2')

    CE, logits_sigmoid = loss.sigmoid_cross_entropy(fc2, label)

    CE_loss = mx.sym.MakeLoss(CE)
    out = mx.sym.Group([CE_loss, mx.sym.BlockGrad(logits_sigmoid)])

    return out


def discriminator_conv(img, label, batch_size, eps, leaky_slope):
    '''
        input image shape should be 64x64xnc
        TODO: make clear the layout of the tensor
    '''
    # TODO: see if fix_gamma = False can also do
    fix_gamma = True
    # 64x64xnc
    conv1 = mx.sym.Convolution(img, num_filter=128, kernel=(4,4), stride=(2,2), pad=(1,1), no_bias=True, name='dis_conv1')
    leaky1 = mx.sym.LeakyReLU(conv1, act_type='leaky', slope=leaky_slope, name='dis_leaky1')
    # 32x32x128
    conv2 = mx.sym.Convolution(leaky1, num_filter=256, kernel=(4,4), stride=(2,2), pad=(1,1), no_bias=True, name='dis_conv2')
    bn1 = mx.sym.BatchNorm(conv2, fix_gamma=fix_gamma, eps=eps, name='dis_bn1')
    leaky2 = mx.sym.LeakyReLU(bn1, act_type='leaky', slope=leaky_slope, name='dis_leaky2')
    # 16x16x256
    conv3 = mx.sym.Convolution(leaky2, num_filter=512, kernel=(4,4), stride=(2,2), pad=(1,1), no_bias=True, name='dis_conv3')
    bn2 = mx.sym.BatchNorm(conv3, fix_gamma=fix_gamma, eps=eps, name='dis_bn2')
    leaky3 = mx.sym.LeakyReLU(bn2, act_type='leaky', slope=leaky_slope, name='dis_leaky3')
    # 8x8x512
    conv4 = mx.sym.Convolution(leaky3, num_filter=1024, kernel=(4,4), stride=(2,2), pad=(1,1), no_bias=True, name='dis_conv4')
    bn3 = mx.sym.BatchNorm(conv4, fix_gamma=fix_gamma, eps=eps, name='dis_bn3')
    leaky4 = mx.sym.LeakyReLU(bn3, act_type='leaky', slope=leaky_slope, name='dis_leaky4')
    # 4x4x1024
    conv5 = mx.sym.Convolution(leaky4, num_filter=1, kernel=(4,4), stride=(1,1), pad=(0,0), no_bias=True, name='dis_conv5')

    logits = mx.sym.Flatten(conv5)

    CE, logits_sigmoid = loss.sigmoid_cross_entropy(logits, label)
    CE_loss = mx.sym.BlockGrad(CE)

    logit_regression = mx.sym.MakeLoss(CE)
    out = mx.sym.Group([CE_loss, mx.sym.BlockGrad(logits_sigmoid), logit_regression])

    #  logit_regression = mx.sym.LogisticRegressionOutput(data=logits, label=label, name='dloss')
    #  out = mx.sym.Group([CE_loss, mx.sym.BlockGrad(logits_sigmoid), logit_regression])

    return out






