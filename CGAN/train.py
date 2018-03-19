#!/usr/bin/python


import mxnet as mx
import numpy as np



batch_size = 128
img_size = 28


mx.random.seed(0)


## symbols
def SoftmaxLoss(data, label):
    softmax = mx.sym.softmax(data, axis=1)
    label_one_hot = mx.sym.one_hot(label, 2)
    ce_mul = mx.sym.sum(label_one_hot * mx.sym.log(softmax), axis=1)
    ce = -mx.sym.mean(ce_mul)

    return ce



def get_gen_cgan():
    fix_gamma = False

    z = mx.sym.var('noise')
    label = mx.sym.var('con_label')
    label_one_hot = mx.sym.one_hot(label, 10).reshape(shape=(batch_size,10,1,1))
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
                                   kernel=(5,5), stride=(2,2),
                                   target_shape=(14,14),
                                   num_filter=128, name='gen_deconv_2')
    bn2 = mx.sym.BatchNorm(deconv2, fix_gamma=fix_gamma, name="gen_bn2")
    lrelu2 = mx.sym.LeakyReLU(bn2, act_type='leaky', slope=0.2,
                              name='gen_leaky2')
    # 128x14x14
    deconv3 = mx.sym.Deconvolution(lrelu2,
                                   kernel=(5,5), stride=(2,2),
                                   target_shape=(28,28),
                                   num_filter=1, name='gen_deconv_3')
    gen_img = mx.sym.tanh(deconv3, name='gen_tanh')

    return gen_img


def get_dis_cgan():
    fix_gamma = False

    img = mx.sym.var('img')
    label = mx.sym.var('label')
    con_label = mx.sym.var('con_label')
    con_label_one_hot = mx.sym.one_hot(con_label, 10).reshape(shape=(batch_size,10,1,1))
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
                               num_filter=2, name='dis_conv3')
    # 2x1x1
    reshape = mx.sym.reshape(conv3, shape=(-1, 2))

    ce_out = mx.sym.SoftmaxOutput(reshape, label)
    Loss = SoftmaxLoss(reshape, label)
    Loss_out = mx.sym.BlockGrad(Loss)

    out = mx.sym.Group([ce_out, Loss_out])

    return out


### module
def get_gen_mod():
    symG = get_gen_cgan()
    modG = mx.mod.Module(symG, context=mx.gpu(),
                         data_names=['noise', 'con_label'],
                         label_names=None)
    modG.bind(data_shapes=[('noise',(batch_size, 100, 1, 1)),
                           ('con_label', (batch_size, 1, 1, 1))])
    modG.init_params(initializer=mx.init.Xavier())
    modG.init_optimizer(optimizer='adam',
                        optimizer_params=(('learning_rate', 2e-4),
                        ('beta1',0.5), ('wd', 0)))

    return modG

def get_dis_mod():
    symD = get_dis_cgan()
    modD = mx.mod.Module(symD, context=mx.gpu(),
                         data_names=['img', 'con_label'],
                         label_names=['label'])
    modD.bind(data_shapes=[('img',(batch_size, 1, img_size, img_size)),
                           ('con_label', (batch_size, 1))],
              label_shapes=[('label', (batch_size,))])
    modD.init_params(initializer=mx.init.Xavier())
    modD.init_optimizer(optimizer='adam',
                        optimizer_params=(('learning_rate', 2e-4),
                        ('beta1', 0.5), ('wd', 0)))

    return modD


### Data Iterator
def trans(data, label):
    #  data = mx.img.imresize(data,64,64)
    data = mx.nd.transpose(data.astype(np.float32)/127.5-1,
                           axes=(2,0,1))
    label.astype(np.float32)
    return data, label

def get_mnist_iter():
    mnist_train = mx.gluon.data.vision.MNIST(root='~/.mxnet/datasets/mnist/',
                                             train=True, transform=trans)
    train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)

    return train_data


## training schema
def train():

    modG = get_gen_mod()
    modD = get_dis_mod()

    trainiter = get_mnist_iter()

    for batch in trainiter:
        noise = mx.nd.random.uniform(0, 1, (batch_size, 100, 1, 1))
        con_label_fake = mx.nd.round(10 * mx.nd.random.uniform(0, 0.9,
                                                   (batch_size,)))

        gen_batch = mx.io.DataBatch(data=[noise, con_label_fake])
        modG.forward(gen_batch, is_train=True)
        img_fake = modG.get_outputs()[0]
        label_fake = mx.nd.zeros(shape=(batch_size,))

        img_real = batch[0].as_in_context(mx.gpu())
        con_label_real = batch[1].as_in_context(mx.gpu())
        label_real = mx.nd.ones(shape=(batch_size,))

        # train disc
        # train on real
        data_batch = mx.io.DataBatch(data=[img_real, con_label_real],
                                     label=[label_real])
        modD.forward(data_batch, is_train=True)
        modD.backward()
        grad_real = [[grad.copyto(grad.context) for grad in grads]
                     for grads in modD._exec_group.grad_arrays]

        data_batch = mx.io.DataBatch(data=[img_fake, con_label_fake],
                                     label=[label_fake])
        # train on fake
        modD.forward(data_batch, is_train=True)
        modD.backward()
        def grad_add(g1, g2):
            g1 += g2
        def grad_list_add(gl1, gl2):
            list(map(grad_add, gl1, gl2))
        list(map(grad_list_add, modD._exec_group.grad_arrays, grad_real))
        modD.update()

        # train gen


        break





if __name__ == '__main__':
    train()
