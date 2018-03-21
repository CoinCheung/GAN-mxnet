#!/usr/bin/python


import mxnet as mx
import symbols.discriminator as discriminator
import symbols.generator as generator
import core.config as config



batch_size = config.batch_size
img_size = config.img_size
lr = config.lr
wd = config.wd
beta1 = config.beta1


def get_gen_mod():
    symG = generator.get_gen_cgan()
    modG = mx.mod.Module(symG, context=mx.gpu(),
                         data_names=['noise', 'con_label'],
                         label_names=None)
    modG.bind(data_shapes=[('noise',(batch_size, 100, 1, 1)),
                           ('con_label', (batch_size, 1, 1, 1))])
    modG.init_params(initializer=mx.init.Xavier())
    lr_schr = mx.lr_scheduler.FactorScheduler(500, 0.95)
    modG.init_optimizer(optimizer='adam',
                        optimizer_params=(('learning_rate', lr),
                            ('beta1', beta1), ('wd', wd),
                            ('lr_scheduler', lr_schr)))

    return modG



def get_dis_mod():
    symD = discriminator.get_dis_cgan()
    modD = mx.mod.Module(symD, context=mx.gpu(),
                         data_names=['img', 'con_label'],
                         label_names=['label'])
    modD.bind(data_shapes=[('img',(batch_size, 1, img_size, img_size)),
                           ('con_label', (batch_size, 1))],
              label_shapes=[('label', (batch_size,))], inputs_need_grad=True)
    modD.init_params(initializer=mx.init.Xavier())
    lr_schr = mx.lr_scheduler.FactorScheduler(500, 0.95)
    modD.init_optimizer(optimizer='adam',
                        optimizer_params=(('learning_rate', lr),
                            ('beta1', beta1), ('wd', wd),
                            ('lr_scheduler', lr_schr)))

    return modD


def get_cgan_mods():
    return get_dis_mod(), get_gen_mod()
