#!/usr/bin/python


import mxnet as mx
import numpy as np
import config
import core.visualize


def generator_conv(noise, batch_size, test, eps):
    '''
    generator of the GAN basing on deconvolution layers.
    The structure is:
        1. fc layer of size 1024 + BN + ReLU
        2. fc of size 7 x 7 x 128 + BN + ReLU
        3. Resize into Image Tensor
        4. transposed conv2 layer with 64 filters of 4x4, stride 2 + BN + ReLU
        5. transposed conv2 layer with 1 filters of 4x4, stride 2 + BN + ReLU
    params:
        noise: a symbol standing for input random noise
        batch_size: the batch size of the generated examples
        test: if the generator works in test mode
        eps: the eps assigned to mx.sym.BatchNorm()
    '''
    '''
    TODO: update innovation
    '''

### old net
    fc1 = mx.sym.FullyConnected(noise, num_hidden=1024, no_bias=False, flatten=True, name='gen_fc1')
    relu1 = mx.sym.Activation(fc1, act_type='relu', name='gen_relu1')
    bn1 = mx.sym.BatchNorm(relu1, fix_gamma=False,use_global_stats=test, eps=eps, name='gen_bn1')

    fc2 = mx.sym.FullyConnected(bn1, num_hidden=7*7*128, no_bias=False, name='gen_fc2')
    relu2 = mx.sym.Activation(fc2, act_type='relu', name='gen_relu2')
    bn2 = mx.sym.BatchNorm(relu2, fix_gamma=False,use_global_stats=test, eps=eps, name='gen_bn2')

    conv_input = bn2.reshape(shape=(batch_size,128,7,7), name='gen_reshape1')

    trans_conv1 = mx.sym.Deconvolution(conv_input, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=64,name="gen_trans_conv1")
    relu3 = mx.sym.Activation(trans_conv1, act_type='relu', name='gen_relu3')
    bn3 = mx.sym.BatchNorm(relu3, fix_gamma=False, use_global_stats=test, eps=eps, name='gen_bn3')

    trans_conv2 = mx.sym.Deconvolution(bn3, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=1,name="gen_trans_conv2")
    tanh1 = mx.sym.Activation(trans_conv2, act_type='tanh', name='gen_tanh1')
###
    #
    #  # 1x1x1
    #  trans_conv1 = mx.sym.Deconvolution(noise, kernel=(3,3), stride=(1,1), pad=(0,0), num_filter=256,name="gen_trans_conv1")
    #  bn1 = mx.sym.BatchNorm(trans_conv1, fix_gamma=False, use_global_stats=test, eps=eps, name='gen_bn1')
    #  relu1 = mx.sym.Activation(bn1, act_type='relu', name='gen_relu1')
    #  # 3x3x256
    #  trans_conv2 = mx.sym.Deconvolution(relu1, kernel=(3,3), stride=(1,1), pad=(0,0), num_filter=256,name="gen_trans_conv2")
    #  bn2 = mx.sym.BatchNorm(trans_conv2, fix_gamma=False, use_global_stats=test, eps=eps, name='gen_bn2')
    #  relu2 = mx.sym.Activation(bn2, act_type='relu', name='gen_relu2')
    #  # 5x5x256
    #  trans_conv3 = mx.sym.Deconvolution(relu2, kernel=(3,3), stride=(1,1), pad=(0,0), num_filter=128,name="gen_trans_conv3")
    #  bn3 = mx.sym.BatchNorm(trans_conv3, fix_gamma=False, use_global_stats=test, eps=eps, name='gen_bn3')
    #  relu3 = mx.sym.Activation(bn3, act_type='relu', name='gen_relu3')
    #  # 7x7x128
    #  trans_conv4 = mx.sym.Deconvolution(relu3, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=64,name="gen_trans_conv4")
    #  bn4 = mx.sym.BatchNorm(trans_conv4, fix_gamma=False, use_global_stats=test, eps=eps, name='gen_bn4')
    #  relu4 = mx.sym.Activation(bn4, act_type='relu', name='gen_relu4')
    #  # 14x14x64
    #  trans_conv5 = mx.sym.Deconvolution(relu4, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=1,name="gen_trans_conv5")
    #  #  bn5 = mx.sym.BatchNorm(trans_conv5, fix_gamma=False, use_global_stats=test, eps=eps, name='gen_bn5')
    #  tanh1 = mx.sym.Activation(trans_conv5, act_type='tanh', name='gen_tanh1')
    #  # 28x28x1

    out = tanh1

    return out




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




if __name__ == "__main__":
    #  out = gen_noise_uniform((1, 4), [-1,1])
    #  print(out)


    eps = config.bn_eps
    test = config.is_test
    batch_size = config.batch_size

    noise = mx.sym.var("noise")

    noise_batch = gen_noise_uniform((batch_size,1024),(-1,1))
    img = generator_conv(noise, batch_size, test, eps)
    gen = mx.mod.Module(img, context=mx.cpu(), data_names=["noise"], label_names=None)
    gen.bind(data_shapes=[('noise',(batch_size,1024))], label_shapes=None)
    gen.init_params()

    batch = mx.io.DataBatch([noise_batch], label=None)
    #  print(batch)

    gen.forward(batch)
    out = gen.get_outputs()[0]
    print(out.asnumpy().shape)
    plot.show_image(out.asnumpy())

