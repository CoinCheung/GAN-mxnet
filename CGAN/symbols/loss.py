#!/usr/bin/python


import mxnet as mx


def SoftmaxLoss(data, label):
    softmax = mx.sym.softmax(data, axis=1)
    label_one_hot = mx.sym.one_hot(label, 2)
    ce_mul = mx.sym.sum(label_one_hot * mx.sym.log(softmax), axis=1)
    ce = -mx.sym.mean(ce_mul)

    return ce


def LogisticLoss(data, label):
    label = label.reshape(shape=(-1, 1))
    sigmoid = mx.sym.sigmoid(data)
    ce_sum = mx.sym.log(sigmoid+1e-12) * label + mx.sym.log(1+1e-12-sigmoid)*(1-label)
    ce = - mx.sym.mean(ce_sum.flatten())

    return ce


