#!/usr/bin/python


import mxnet as mx
import core.config as config



batch_size = config.batch_size


def get_gen_cgan():
    fix_gamma = False

    z = mx.sym.var('noise')
    label = mx.sym.var('con_label')
    label_one_hot = mx.sym.one_hot(label, 10).reshape(shape=(batch_size,10,1,1))
    yz = mx.sym.concat(z, label_one_hot, dim=1)
    # 110x1x1
    deconv1 = mx.sym.Deconvolution(yz,
                                   kernel=(7,7), stride=(1,1),
                                   num_filter=256, name='gen_deconv_1')
    bn1 = mx.sym.BatchNorm(deconv1, fix_gamma=fix_gamma, name="gen_bn1")
    lrelu1 = mx.sym.Activation(bn1, act_type='relu', name='gen_relu1')
    # 256x7x7
    deconv2 = mx.sym.Deconvolution(lrelu1,
                                   kernel=(5,5), stride=(2,2),
                                   target_shape=(14,14),
                                   num_filter=128, name='gen_deconv_2')
    bn2 = mx.sym.BatchNorm(deconv2, fix_gamma=fix_gamma, name="gen_bn2")
    lrelu2 = mx.sym.Activation(bn2, act_type='relu', name='gen_relu2')
    # 128x14x14
    deconv3 = mx.sym.Deconvolution(lrelu2,
                                   kernel=(5,5), stride=(2,2),
                                   target_shape=(28,28),
                                   num_filter=1, name='gen_deconv_3')
    gen_img = mx.sym.tanh(deconv3, name='gen_tanh')

    return gen_img

