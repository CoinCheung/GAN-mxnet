#!/usr/bin/python


import mxnet as mx
import numpy as np
import os
import core.visualize as visualize



batch_size = 128
img_size = 28
epoch = 5

lr = 2e-4
beta1 = 0.5
wd = 0

mx.random.seed(0)


## symbols
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
    lrelu1 = mx.sym.Activation(bn1, act_type='relu', name='gen_relu1')
    # 256x7x7
    deconv2 = mx.sym.Deconvolution(lrelu1,
                                   kernel=(5,5), stride=(2,2),
                                   target_shape=(14,14),
                                   num_filter=128, name='gen_deconv_2')
    bn2 = mx.sym.BatchNorm(deconv2, fix_gamma=fix_gamma, name="gen_bn2")
    lrelu2 = mx.sym.Activation(bn2, act_type='relu', name='gen_relu2')
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
    #  conv3 = mx.sym.Convolution(lrelu2, kernel=(7,7), stride=(1,1),
    #                             num_filter=2, name='dis_conv3')
    # 2x1x1
    #  scores = mx.sym.reshape(conv3, shape=(-1, 2))
    #  ce_out = mx.sym.SoftmaxOutput(scores, label)
    #  Loss = SoftmaxLoss(scores, label)

    conv3 = mx.sym.Convolution(lrelu2, kernel=(7,7), stride=(1,1),
                               num_filter=1, name='dis_conv3')
    # 2x1x1
    scores = mx.sym.flatten(conv3)

    ce_out = mx.sym.LogisticRegressionOutput(scores, label)
    Loss = LogisticLoss(scores, label)
    Loss_out = mx.sym.BlockGrad(Loss)
    scores_out = mx.sym.BlockGrad(mx.sym.sigmoid(scores))

    out = mx.sym.Group([ce_out, Loss_out, scores_out])

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
    lr_schr = mx.lr_scheduler.FactorScheduler(500, 0.95)
    modG.init_optimizer(optimizer='adam',
                        optimizer_params=(('learning_rate', lr),
                            ('beta1', beta1), ('wd', wd),
                            ('lr_scheduler', lr_schr)))

    return modG

def get_dis_mod():
    symD = get_dis_cgan()
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


### Data Iterator
def trans(data, label):
    #  data = mx.img.imresize(data,64,64)
    data = mx.nd.transpose(data.astype(np.float32)/127.5-1, axes=(2,0,1))
    label.astype(np.float32)
    return data, label

def get_mnist_iter():
    mnist_train = mx.gluon.data.vision.MNIST(root='~/.mxnet/datasets/mnist/',
                                             train=True, transform=trans)
    train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)

    return train_data


## accuracy
def real_ratio(scores):
    #  cls = np.argmax(scores, axis=1)
    cls = np.round(scores).reshape((-1,))
    return np.mean(cls)



## training schema
def train():

    save_path = ''.join([os.getcwd(), '/model_export/'])

    modG = get_gen_mod()
    modD = get_dis_mod()

    train_iter = get_mnist_iter()

    img_handle = visualize.ImgGrids(1)

    iter_count = 0
    for e in range(epoch):
        for batch in train_iter:
            if batch[0].shape[0] != batch_size:
                continue

            noise = mx.nd.random.uniform(-1, 1, (batch_size, 100, 1, 1))
            con_label = batch[1].as_in_context(mx.gpu())

            gen_batch = mx.io.DataBatch(data=[noise, con_label])
            modG.forward(gen_batch, is_train=True)
            img_fake = modG.get_outputs()[0]
            label_fake = mx.nd.zeros(shape=(batch_size,))

            img_real = batch[0].as_in_context(mx.gpu())
            label_real = mx.nd.ones(shape=(batch_size,))

            # train disc
            # train on real
            data_batch = mx.io.DataBatch(data=[img_real, con_label],
                                         label=[label_real])
            modD.forward(data_batch, is_train=True)
            LossD_real = modD.get_outputs()[1]
            scores_real = modD.get_outputs()[2].asnumpy()
            modD.backward()
            grad_real = [[grad.copyto(grad.context) for grad in grads]
                         for grads in modD._exec_group.grad_arrays]

            # train on fake
            data_batch = mx.io.DataBatch(data=[img_fake, con_label],
                                         label=[label_fake])
            modD.forward(data_batch, is_train=True)
            LossD_fake = modD.get_outputs()[1]
            scores_fake = modD.get_outputs()[2].asnumpy()
            modD.backward()
            def grad_add(g1, g2):
                g1 += g2
                g1 /= 2
            def grad_list_add(gl1, gl2):
                list(map(grad_add, gl1, gl2))
            list(map(grad_list_add, modD._exec_group.grad_arrays, grad_real))
            modD.update()


            # train gen
            data_batch = mx.io.DataBatch(data=[img_fake, con_label],
                                         label=[label_real])
            modD.forward(data_batch, is_train=True)
            lossG = modD.get_outputs()[1]
            modD.backward()
            grad_for_G = modD.get_input_grads()
            modG.backward(grad_for_G)
            modG.update()

            iter_count += 1
            if iter_count % 50 == 0:
                real_prob = real_ratio(scores_real)
                fake_prob = real_ratio(scores_fake)
                lossD = ((LossD_real + LossD_fake) / 2).asnumpy()
                lossG = lossG.asnumpy()
                print("lossG: {}, lossD: {}, real acc: {}, fake acc: {}".format(lossG, lossD, real_prob, fake_prob))
                img_handle.draw(img_fake.asnumpy()[:64])
        print("epoch {} accomplished!".format(e))

    modD.save_checkpoint(save_path+'modD', 0, True)
    modG.save_checkpoint(save_path+'modG', 0, True)





if __name__ == '__main__':
    train()
    a = input()
