#!/usr/bin/python


import mxnet as mx
import symbol.discriminator
import symbol.generator
import core.DataIter
import core.config as config



def softmax(pred):
    '''
    compute the softmax probability given a predicted array

    params:
        pred: the output scores of some network, of shape [batch_size, n]
    '''
    pred_exp = mx.sym.exp(pred)
    return mx.sym.broadcast_div(pred_exp, mx.sym.sum(pred_exp, axis=1, keepdims=True))



def softmax_cross_entropy_binary(pred, label, batch_size):
    '''
    a method to compute the binary softmax cross entropy given the network output
    scores of the two classes and their associated real labels.

    params:
        pred: the binary output scores of a network which will be softmaxed
        label: the one-number label of each scores.
    '''
    label = mx.sym.reshape(label, shape=(-1,))
    label = mx.sym.one_hot(label, 2)
    pred_prob = softmax(pred)
    pred_prob_log = mx.sym.log(pred_prob)
    product = mx.sym.broadcast_mul(pred_prob_log, label, name='product')
    CE = -mx.sym.sum(product)
    return out/batch_size


def sigmoid_cross_entropy(logits, label, batch_size):
    '''
    used in binary case naturally
    '''
    logits_sigmoid = mx.sym.sigmoid(logits).reshape(shape=(-1, 1))
    logits_sigmoid_log_real = mx.sym.log(logits_sigmoid)
    logits_sigmoid_log_fake = mx.sym.log(1-logits_sigmoid)
    product = logits_sigmoid_log_real*label + logits_sigmoid_log_fake*(1-label)
    CE = -mx.sym.sum(product)/batch_size

    return [CE, logits_sigmoid]



def gan_loss(real_batch, fake_batch, batch_size):
    '''
    compute the softmax cross entropy loss of a gan. All symbols along the path
    to the module sym should not overlap with their names. Thus for discriminator
    loss, the discriminator_conv() method should not be reused, and the input
    symbols should be concated instead.

    params:
        real_batch: a symbol of the real image data batch
        fake_batch: a symbol of the fake image data generated from noise
        batch_size: the batch size of the two batches (assume they have
        identical batch size)
    '''
    # discriminator loss
    batch_all = mx.sym.concat(real_batch, fake_batch, dim=0)
    dis_all = discriminator.discriminator_conv(batch_all, batch_size)

    label_dis_real = mx.sym.ones((batch_size, 1))
    label_dis_fake = mx.sym.zeros((batch_size, 1))
    label_dis_all = mx.sym.concat(label_dis_real, label_dis_fake, dim=0)

    #  loss_D = softmax_cross_entropy_binary(dis_all, label_dis_all, 2*batch_size)
    loss_D = sigmoid_cross_entropy(dis_all, label_dis_all, 2*batch_size)
    loss_D = mx.sym.MakeLoss(loss_D)

    out = mx.sym.BlockGrad(dis_all)

    loss_D = mx.sym.Group([loss_D, out])


    # generator loss
    dis_gen = discriminator.discriminator_conv(fake_batch, batch_size)
    label_gen = mx.sym.ones((batch_size, 1))
    #  loss_G = softmax_cross_entropy_binary(dis_gen, label_gen, batch_size)
    loss_G = sigmoid_cross_entropy(dis_gen, label_gen, batch_size)
    loss_G = mx.sym.MakeLoss(loss_G)

    return loss_D, loss_G



if __name__ == "__main__":


    # params
    eps = config.bn_eps
    test = config.is_test
    batch_size = config.batch_size
    img_size = config.img_size

    # syms
    real_batch = mx.sym.var('real_batch')
    noise = mx.sym.var("noise")
    fake_batch = generator.generator_conv(noise, batch_size, test, eps)
    loss_D, loss_G = gan_loss(real_batch, fake_batch, batch_size)

    # datas iters
    it = IO.get_mnist_iter()

    # modules
    gen = mx.mod.Module(loss_G, context=mx.cpu(), data_names=['noise'], label_names=None)
    gen.bind(data_shapes=[('noise',(batch_size, 1024))], label_shapes=None)
    gen.init_params(mx.initializer.Xavier())

    dis = mx.mod.Module(loss_D, context=mx.cpu(), data_names=['real_batch', 'noise'], label_names=None)
    dis.bind(data_shapes=[('real_batch',(batch_size,1,img_size,img_size)), ('noise', (batch_size,1024))], label_shapes=None)
    dis.init_params(mx.initializer.Xavier())

    # compute loss values
    for img,label in it:
        noise_array = generator.gen_noise_uniform((batch_size,1024),(-1,1))
        dis_batch = mx.io.DataBatch([img, noise_array], label=None)
        gen_batch = mx.io.DataBatch([noise_array], label=None)

        gen.forward(gen_batch)
        dis.forward(dis_batch)

        lossG = gen.get_outputs()[0]
        lossD = dis.get_outputs()[0]

        print(lossG.asnumpy())
        print(lossD.asnumpy())

        #
        temp = dis.get_outputs()[1].asnumpy().shape
        print(temp)


        break




