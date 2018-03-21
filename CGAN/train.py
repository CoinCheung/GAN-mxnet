#!/usr/bin/python


import mxnet as mx
import numpy as np
import os
import core.visualize as visualize
import core.modules as modules
import core.config as config
import core.DataIter as DI



# hyper-parameters
mx.random.seed(0)
batch_size = config.batch_size
epoch = config.epoch
save_path = os.getcwd() + config.save_path



## accuracy
def real_ratio(scores):
    #  cls = np.argmax(scores, axis=1)
    cls = np.round(scores).reshape((-1,))
    return np.mean(cls)



## training schema
def train():

    modD, modG = modules.get_cgan_mods()

    train_iter = DI.get_mnist_iter()

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
    os.mkdir(save_path)
    train()
    a = input()
