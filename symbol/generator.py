#!/usr/bin/python


import mxnet as mx
import core.config as config


def generator_lenet5(noise, batch_size, img_channels, eps):
    '''
    generator of the GAN basing on deconvolution layers.
    The structure is:
        1. fc layer of size 1024 + BN + ReLU
        2. fc of size 8 x 8 x 128 + BN + ReLU
        3. Resize into Image Tensor
        4. transposed conv2 layer with 64 filters of 4x4, stride 2, pad 1 + BN + ReLU
        5. transposed conv2 layer with 1 filters of 4x4, stride 2, pad 1 + BN + ReLU
    params:
        noise: a symbol standing for input random noise
        batch_size: the batch size of the generated examples
        eps: the eps assigned to mx.sym.BatchNorm()
    '''

    fix_gamma = True

    fc1 = mx.sym.FullyConnected(noise, num_hidden=1024, no_bias=False, flatten=True, name='gen_fc1')
    relu1 = mx.sym.Activation(fc1, act_type='relu', name='gen_relu1')
    bn1 = mx.sym.BatchNorm(relu1, fix_gamma=fix_gamma, eps=eps, name='gen_bn1')

    fc2 = mx.sym.FullyConnected(bn1, num_hidden=8*8*128, no_bias=False, name='gen_fc2')
    relu2 = mx.sym.Activation(fc2, act_type='relu', name='gen_relu2')
    bn2 = mx.sym.BatchNorm(relu2, fix_gamma=fix_gamma, eps=eps, name='gen_bn2')

    conv_input = bn2.reshape(shape=(batch_size,128,8,8), name='gen_reshape1')
    # 128x8x8
    trans_conv1 = mx.sym.Deconvolution(conv_input, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=64,name="gen_trans_conv1")
    relu3 = mx.sym.Activation(trans_conv1, act_type='relu', name='gen_relu3')
    bn3 = mx.sym.BatchNorm(relu3, fix_gamma=fix_gamma, eps=eps, name='gen_bn3')
    # 64x16x16
    trans_conv2 = mx.sym.Deconvolution(bn3, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=32,name="gen_trans_conv2")
    relu4 = mx.sym.Activation(trans_conv2, act_type='relu', name='gen_relu4')
    bn4 = mx.sym.BatchNorm(relu4, fix_gamma=fix_gamma, eps=eps, name='gen_bn4')
    # 32x32x32
    trans_conv3 = mx.sym.Deconvolution(bn4, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=img_channels,name="gen_trans_conv3")
    tanh1 = mx.sym.Activation(trans_conv3, act_type='tanh', name='gen_tanh1')
    # ncx64x64

    out = tanh1

    return out


def generator_fcn(noise, batch_size, nc, eps):
    '''
        symbol noise has a shape of batch_sizexncx1x1
    '''
    fix_gamma = True
    # 1x1x100
    trans_conv1 = mx.sym.Deconvolution(noise, kernel=(4,4), stride=(1,1), pad=(0,0), num_filter=1024, name='gen_trans_conv1')
    bn1 = mx.sym.BatchNorm(trans_conv1, fix_gamma=fix_gamma, eps=eps, name='gen_bn1')
    relu1 = mx.sym.Activation(bn1, act_type='relu', name='gen_relu1')
    # 4x4x1024
    trans_conv2 = mx.sym.Deconvolution(relu1, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=512, name='gen_trans_conv2')
    bn2 = mx.sym.BatchNorm(trans_conv2, fix_gamma=fix_gamma, eps=eps, name='gen_bn2')
    relu2 = mx.sym.Activation(bn2, act_type='relu', name='gen_relu2')
    # 8x8x512
    trans_conv3 = mx.sym.Deconvolution(relu2, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, name='gen_trans_conv3')
    bn3 = mx.sym.BatchNorm(trans_conv3, fix_gamma=fix_gamma, eps=eps, name='gen_bn3')
    relu3 = mx.sym.Activation(bn3, act_type='relu', name='gen_relu3')
    # 16x16x256
    trans_conv4 = mx.sym.Deconvolution(relu3, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, name='gen_trans_conv4')
    bn4 = mx.sym.BatchNorm(trans_conv4, fix_gamma=fix_gamma, eps=eps, name='gen_bn4')
    relu4 = mx.sym.Activation(bn4, act_type='relu', name='gen_relu4')
    # 32x32x128
    trans_conv5 = mx.sym.Deconvolution(relu4, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, name='gen_trans_conv5')
    # 64x64xnc

    tanh = mx.sym.Activation(trans_conv5, act_type='tanh', name='gen_output')

    return tanh





