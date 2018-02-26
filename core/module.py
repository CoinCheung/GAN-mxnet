#!/usr/bin/python


import mxnet as mx
import symbol.discriminator as discriminator
import symbol.generator as generator
import core.config as config


def get_usual_modules():
    # parameters
    eps = config.bn_eps
    batch_size = config.batch_size
    noise_shape = config.noise_shape
    img_shape = config.img_shape
    lky_slope = config.leaky_slope
    opt_gen = config.gen_optimizer
    opt_dis = config.dis_optimizer
    gen_optimizer_params = config.gen_optimizer_params
    dis_optimizer_params = config.dis_optimizer_params
    Gan_type = config.GAN_type
    nc = img_shape[1]

    ## syms
    img = mx.sym.var('image')
    label = mx.sym.var('label')
    noise = mx.sym.var("noise")
    gen_batch_sym = generator.generator_usual(noise, batch_size, nc, eps)
    dis_sigmoid_loss = discriminator.discriminator_usual(img, label, batch_size*2, lky_slope)
    ## modules
    # generator
    gen = mx.mod.Module(gen_batch_sym, context=mx.gpu(), data_names=['noise'], label_names=None)
    gen.bind(data_shapes=[('noise',noise_shape)], label_shapes=None, inputs_need_grad=True)
    gen.init_params(initializer=mx.initializer.Xavier())
    gen.init_optimizer(optimizer=opt_gen, optimizer_params=gen_optimizer_params)

    # discriminator
    dis = mx.mod.Module(dis_sigmoid_loss, context=mx.gpu(), data_names=['image'], label_names=['label'])
    dis.bind(data_shapes=[('image',(batch_size*2, nc, 64, 64))], label_shapes=[('label',(batch_size*2,1))], inputs_need_grad=True)
    dis.init_params(initializer=mx.initializer.Xavier())
    dis.init_optimizer(optimizer=opt_dis, optimizer_params=dis_optimizer_params)

    return gen, dis

def get_dc_modules():
    # parameters
    eps = config.bn_eps
    batch_size = config.batch_size
    noise_shape = config.noise_shape
    img_shape = config.img_shape
    lky_slope = config.leaky_slope
    opt_gen = config.gen_optimizer
    opt_dis = config.dis_optimizer
    gen_optimizer_params = config.gen_optimizer_params
    dis_optimizer_params = config.dis_optimizer_params
    nc = img_shape[1]

    ## syms
    img = mx.sym.var('image')
    label = mx.sym.var('label')
    noise = mx.sym.var("noise")
    gen_batch_sym = generator.generator_conv(noise, batch_size, nc, eps)
    dis_sigmoid_loss = discriminator.discriminator_conv(img, label, batch_size*2, eps, lky_slope)
    ## modules
    # generator
    gen = mx.mod.Module(gen_batch_sym, context=mx.gpu(), data_names=['noise'], label_names=None)
    gen.bind(data_shapes=[('noise',noise_shape)], label_shapes=None, inputs_need_grad=True)
    gen.init_params(initializer=mx.init.Normal(0.02))
    #  gen.init_params(initializer=mx.init.Xavier())
    gen.init_optimizer(optimizer=opt_gen, optimizer_params=gen_optimizer_params)

    # discriminator
    dis = mx.mod.Module(dis_sigmoid_loss, context=mx.gpu(), data_names=['image'], label_names=['label'])
    dis.bind(data_shapes=[('image',(batch_size*2, nc, 64, 64))], label_shapes=[('label',(batch_size*2,1))], inputs_need_grad=True)
    dis.init_params(initializer=mx.init.Normal(0.02))
    #  dis.init_params(initializer=mx.init.Xavier())
    dis.init_optimizer(optimizer=opt_dis, optimizer_params=dis_optimizer_params)

    return gen, dis


def get_modules():
    # control params
    Gan_type = config.GAN_type

    if Gan_type == 'usual_gan':
        return get_usual_modules()

    elif Gan_type == 'dc_gan':
        return get_dc_modules()




