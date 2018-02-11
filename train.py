#!/usr/bin/python


import mxnet as mx
import numpy as np
import os
import symbol.generator as generator
import config
import core.visualize as visualize
import record.save as save
import core.module as module
import core.meric as meric
import core.DataIter as DI




def train_gan(draw=True):
    ## control params
    epoch = config.epoch
    batch_size = config.batch_size
    noise_shape = config.noise_shape
    save_epoch_num = config.save_each_epoch
    save_path = ''.join([os.getcwd(), '/record/'])


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
            noise_array = DI.gen_noise_uniform(noise_shape,(-1,1))
            noise_batch = mx.io.DataBatch([noise_array], label=None)
            gen.forward(noise_batch)
            img_batch_real = img.as_in_context(mx.gpu())
            img_batch_fake = gen.get_outputs()[0]
            label_real = mx.nd.ones((batch_size, 1), ctx=mx.gpu())
            label_fake = mx.nd.zeros((batch_size, 1), ctx=mx.gpu())
            img_batch = mx.nd.concat(img_batch_real, img_batch_fake, dim=0)
            label_batch = mx.nd.concat(label_real, label_fake, dim=0)
            dis_batch = mx.io.DataBatch([img_batch], label=[label_batch])
            # forward and backward
            dis.forward(dis_batch)
            dis.backward()
            dis.update()
            lossD.append(dis.get_outputs()[0].asnumpy())
            out_D = dis.get_outputs()

            ## train generator
            noise_array = DI.gen_noise_uniform(noise_shape,(-1,1))
            noise_batch = mx.io.DataBatch([noise_array], label=None)
            # generator forward
            gen.forward(noise_batch)
            gen_output = gen.get_outputs()[0]
            label = mx.nd.ones((batch_size,1), ctx=mx.gpu())
            gen_batch = mx.io.DataBatch([gen_output], label=[label])

            #  gen_batch = mx.io.DataBatch([img_batch_fake], label=[label])
            dis.forward(gen_batch)
            dis.backward()
            grad = dis.get_input_grads()[0]
            gen.backward([grad])
            gen.update()
            lossG.append(dis.get_outputs()[0].asnumpy())

            i += 1
            if i % 50 == 0:
                lossD_val = lossD[-1]
                lossG_val = lossG[-1]
                Dacc_real, Dacc_fake = meric.dis_acc(out_D[1].asnumpy(), batch_size)
                print("epoch {}, iter {}, lossD {}, lossG {}, Dacc_real {}, Dacc_fake {}".format(e,i,np.round(lossD_val, 4), np.round(lossG_val, 4), Dacc_real, Dacc_fake))
                if draw and i % 200 == 0:
                    img_data = np.ceil((gen.get_outputs()[0].asnumpy()+1)*128).astype(np.uint8)
                    ImgHandler.draw(img_data[:64])

        ## save params
        if e != 0 and e % save_epoch_num == 0:
            gen.save_checkpoint(save_path+'gen', 0, True)
            dis.save_checkpoint(save_path+'dis', 0, True)

    visualize.draw_loss([np.array(lossD),np.array(lossG)], ['lossD','lossG'], 2)




if __name__ == "__main__":
    train_gan(draw=True)



