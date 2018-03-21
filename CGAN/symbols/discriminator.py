#!/usr/bin/python


import mxnet as mx
import symbols.loss as loss
import core.config as config


batch_size = config.batch_size
img_size = config.img_size



def get_dis_cgan():
    fix_gamma = False

    img = mx.sym.var('img')
    label = mx.sym.var('label')
    con_label = mx.sym.var('con_label')
    con_label_one_hot = mx.sym.one_hot(con_label,
                                       10).reshape(shape=(batch_size,10,1,1))
    label_con = mx.sym.broadcast_mul(con_label_one_hot,
                        mx.sym.ones((batch_size, 10, img_size, img_size)))
    img_concat = mx.sym.concat(img, label_con, dim=1)

    # 11x28x28
    conv1 = mx.sym.Convolution(img_concat, kernel=(5,5), stride=(2,2),
                               pad=(2,2), num_filter=128, name='dis_conv1')
    lrelu1 = mx.sym.LeakyReLU(conv1, act_type='leaky', slope=0.2,
                              name='dis_leaky1')
    # 128x14x14
    conv2 = mx.sym.Convolution(lrelu1, kernel=(5,5), stride=(2,2),
                               pad=(2,2), num_filter=256, name='dis_conv2')
    bn2 = mx.sym.BatchNorm(conv2, fix_gamma=fix_gamma, name='dis_bn2')
    lrelu2 = mx.sym.LeakyReLU(bn2, act_type='leaky', slope=0.2,
                              name='dis_leaky2')
    # 256x7x7
    conv3 = mx.sym.Convolution(lrelu2, kernel=(7,7), stride=(1,1),
                               num_filter=1, name='dis_conv3')
    # 1x1x1
    scores = mx.sym.flatten(conv3)

    ce_out = mx.sym.LogisticRegressionOutput(scores, label)
    Loss = loss.LogisticLoss(scores, label)
    Loss_out = mx.sym.BlockGrad(Loss)
    scores_out = mx.sym.BlockGrad(mx.sym.sigmoid(scores))

    out = mx.sym.Group([ce_out, Loss_out, scores_out])

    return out
