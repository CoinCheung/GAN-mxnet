#!/usr/bin/python


import numpy as np


def dis_acc(logits_sigmoid, batch_size):
    cls = np.round(logits_sigmoid).reshape(-1,1)
    cls_real = cls[:batch_size,:]
    cls_fake = cls[batch_size:,:]
    #  print(logits_sigmoid.reshape((-1,)).shape[0])
    real_label = np.ones((batch_size,1))
    fake_label = np.zeros((batch_size,1))
    acc_real = 100*np.sum(cls_real==real_label)/cls_real.shape[0]
    acc_fake = 100*np.sum(cls_fake==fake_label)/cls_fake.shape[0]

    return acc_real, acc_fake




