#!/usr/bin/python


import mxnet as mx



def softmax(pred):
    '''
    compute the softmax probability given a predicted array

    params:
        pred: the output scores of some network, of shape [batch_size, n]
    '''
    pred_exp = mx.sym.exp(pred-mx.sym.max(pred, axis=1))
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


def sigmoid_cross_entropy(logits, label):
    '''
    used in binary case naturally
    '''

    logits_sigmoid = mx.sym.sigmoid(logits).reshape(shape=(-1, 1))

    product = mx.sym.relu(logits) - logits*label + mx.sym.log(1+mx.sym.exp(-mx.sym.abs(logits)) )
    #  product = logits - logits*label - mx.sym.log(mx.sym.sigmoid(logits) + (1e-12))
# TODO: see if sigmoid adding max and abs can do, see if no 1e-12 can do

# without 1e-12, the distracted model can still be converged after some iters
# training. But the generated characters are not clear

    CE = mx.sym.mean(product)

    return [CE, logits_sigmoid]




