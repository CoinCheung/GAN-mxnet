#!/usr/bin/python


import numpy as np


def dis_acc(sigmoid_true, sigmoid_fake):
    sigmoid_true = sigmoid_true.reshape((-1,))
    sigmoid_fake = sigmoid_fake.reshape((-1,))

    pred_true = np.round(sigmoid_true)
    pred_fake = np.round(sigmoid_fake)

    acc_real = 100*np.sum(pred_true)/pred_true.shape[0]
    acc_fake = 100*np.sum(1-pred_fake)/pred_fake.shape[0]

    return acc_real, acc_fake




