#!/usr/bin/python


import mxnet as mx
import numpy as np
import os
import symbol.generator as generator
import core.config as config
import core.visualize as visualize
import core.module as module
import core.meric as meric
import core.DataIter as DI




def train_gan(draw=True):
    ## control params
    epoch = config.epoch
    batch_size = config.batch_size
    noise_shape = config.noise_shape
    save_epoch_num = config.save_each_epoch
    save_path = ''.join([os.getcwd(), '/model_export/'])
    print_iters = 50
    draw_iters = 50


    ## get modules
    gen, dis = module.get_modules()

    ## visualizing
    ImgHandler = visualize.ImgGrids(1)

    ## training procedure
    lossD = []
    lossG = []
    it = DI.get_mnist_iter()
    i = 0
    for e in range(epoch):
        for img,label in it:
            if img.shape[0] != batch_size:
                continue

            ## train discriminator
            # generate fake images
            noise_array = DI.gen_noise_uniform(noise_shape, (-1,1))
            gen.forward(mx.io.DataBatch([noise_array], label=None))
            fake_images = gen.get_outputs()[0]

            # train discriminator with fake images without update, and store grad
            label_fake = mx.nd.zeros((batch_size, 1), ctx=mx.gpu())
            dis.forward(mx.io.DataBatch([fake_images], label=[label_fake]))
            dis.backward()
            grad_fake = [[grad.copyto(grad.context) for grad in grads] for grads in dis._exec_group.grad_arrays]
            # output information
            lossD_fake = dis.get_outputs()[0]
            if i % print_iters  == 0:
                sigmoid_logits_fake = dis.get_outputs()[1].asnumpy()

            # train discriminator with true images and add fake grads
            label_true = mx.nd.ones((batch_size, 1), ctx=mx.gpu())
            dis.forward(mx.io.DataBatch([img.as_in_context(mx.gpu())], label=[label_true]))
            dis.backward()

            # add the fake grads and true grades together
            def grad_add(g1, g2):
                g1 += g2
            def grad_list_add(gl1, gl2):
                list(map(grad_add, gl1, gl2))
            list(map(grad_list_add, dis._exec_group.grad_arrays, grad_fake))
            # update
            dis.update()
            # get output information to print
            lossD_true = dis.get_outputs()[0]
            lossD.append((lossD_true+lossD_fake).asnumpy()/2)
            if i % print_iters  == 0:
                sigmoid_logits_true = dis.get_outputs()[1].asnumpy()


            ## train generator
            dis.forward(mx.io.DataBatch([fake_images], label=[label_true]))
            dis.backward()
            grad_gen = dis.get_input_grads()
            gen.backward(grad_gen)
            gen.update()
            lossG.append(dis.get_outputs()[0].asnumpy())


            i += 1
            if i % print_iters == 0:
                lossD_val = lossD[-1]
                lossG_val = lossG[-1]
                Dacc_true, Dacc_fake = meric.dis_acc(sigmoid_logits_true, sigmoid_logits_fake)
                print("epoch {}, iter {}, lossD {}, lossG {}, Dacc_true {}, Dacc_fake {}".format(e,i,np.round(lossD_val, 4), np.round(lossG_val, 4), Dacc_true, Dacc_fake))

            if draw and (i%draw_iters==0):
                img_data = np.ceil((fake_images.asnumpy()+1)*127.5).astype(np.uint8)
                ImgHandler.draw(img_data[:64])

        ## save params
        if e != 0 and e % save_epoch_num == 0:
            gen.save_checkpoint(save_path+'gen', 0, True)
            dis.save_checkpoint(save_path+'dis', 0, True)

    visualize.draw_loss([np.array(lossD),np.array(lossG)], ['lossD','lossG'], 2)




if __name__ == "__main__":
    train_gan(draw=True)



