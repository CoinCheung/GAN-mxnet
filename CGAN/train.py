#!/usr/bin/python


import mxnet as mx



batch_size = 128
img_size = 28


mx.random.seed(0)


## symbols
def get_gen_cgan():
    fix_gamma = False

    z = mx.sym.random.uniform(0, 1, (batch_size, 100, 1, 1))
    label = mx.sym.round(10 * mx.sym.random.uniform(0, 0.9,
                                               (batch_size, 10, 1, 1)))
    label_one_hot = mx.sym.one_hot(label, 10).reshape(batch_size,10,1,1)
    yz = mx.sym.concat(z, label_one_hot, dim=1)
    # 110x1x1
    deconv1 = mx.sym.Deconvolution(yz,
                                   kernel=(7,7), stride=(1,1),
                                   num_filter=256, name='gen_deconv_1')
    bn1 = mx.sym.BatchNorm(deconv1, fix_gamma=fix_gamma, name="gen_bn1")
    lrelu1 = mx.sym.LeakyReLU(bn1, act_type='leaky', slope=0.2,
                              name='gen_leaky1')
    # 256x7x7
    deconv2 = mx.sym.Deconvolution(lrelu1,
                                   kernel=(5,5), stride=(2,2), pad=(2,2),
                                   num_filter=128, name='gen_deconv_2')
    bn2 = mx.sym.BatchNorm(deconv2, fix_gamma=fix_gamma, name="gen_bn2")
    lrelu2 = mx.sym.LeakyReLU(bn2, act_type='leaky', slope=0.2,
                              name='gen_leaky2')
    # 128x14x14
    deconv3 = mx.sym.Deconvolution(lrelu2,
                                   kernel=(5,5), stride=(2,2), pad=(2,2),
                                   num_filter=1, name='gen_deconv_3')
    gen_img = mx.sym.tanh(deconv3, name='gen_tanh')

    return gen_img, label


def get_dis_cgan():
    pass





















