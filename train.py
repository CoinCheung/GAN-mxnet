#!/usr/bin/python


import mxnet as mx
import numpy as np
import os
import symbol.discriminator as discriminator
import symbol.generator as generator
import core.IO as IO
import config
import core.visualize as visualize
import record.save as save




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




def train_gan(draw=True):
    ## control params
    train_gen_iternums = config.train_gen_iternums
    train_dis_iternums = config.train_dis_iternums
    epoch = config.epoch
    eps = config.bn_eps
    test = config.is_test
    batch_size = config.batch_size
    channel_num = config.img_channel
    img_size = config.img_size
    save_epoch_num = config.save_each_epoch
    save_path = ''.join([os.getcwd(), '/record/'])
    lr_dis = config.dis_learning_rate
    lr_gen = config.gen_learning_rate
    opt_gen = config.gen_optimizer
    opt_dis = config.dis_optimizer
    dis_wd = config.dis_weight_decay
    gen_wd = config.gen_weight_decay
    lky_slope = config.leaky_slope
    noise_shape = config.noise_shape


    ## syms
    img = mx.sym.var('image')
    label = mx.sym.var('label')
    noise = mx.sym.var("noise")
    gen_batch_sym = generator.generator_conv(noise, batch_size, test, eps)
    dis_sigmoid_loss = discriminator.discriminator_conv(img, label, batch_size, test, lky_slope, eps)


    ## modules
    # generator
    gen = mx.mod.Module(gen_batch_sym, context=mx.gpu(), data_names=['noise'], label_names=None)
    gen.bind(data_shapes=[('noise',noise_shape)], label_shapes=None, inputs_need_grad=True)
    gen.init_params(initializer=mx.initializer.Xavier())
    gen.init_optimizer(optimizer=opt_gen, optimizer_params=(('learning_rate', lr_gen), ('beta1',0.5), ('wd', gen_wd)))

    # discriminator
    dis = mx.mod.Module(dis_sigmoid_loss, context=mx.gpu(), data_names=['image'], label_names=['label'])
    dis.bind(data_shapes=[('image',(batch_size, channel_num, img_size, img_size))], label_shapes=[('label',(batch_size,1))], inputs_need_grad=True)
    dis.init_params(initializer=mx.initializer.Xavier())
    dis.init_optimizer(optimizer=opt_dis, optimizer_params=(('learning_rate',lr_dis),('beta1', 0.5),('wd', dis_wd)))


    ## visualizing
    ImgHandler = visualize.ImgGrids(1)


    ## training procedure
    lossD = []
    lossG = []
    it = IO.get_mnist_iter()
    i = 0
    for e in range(epoch):
        for img,label in it:
            if img.shape[0] != batch_size:
                continue

            ## train discriminator
            for dis_num in range(train_dis_iternums):
                noise_array = generator.gen_noise_uniform(noise_shape,(-1,1))
                noise_batch = mx.io.DataBatch([noise_array], label=None)
                gen.forward(noise_batch)
                img_batch_real = img.as_in_context(mx.gpu())
                img_batch_fake = gen.get_outputs()[0]
                label_real = mx.nd.ones((batch_size, 1))
                label_fake = mx.nd.zeros((batch_size, 1))
                img_batch = mx.nd.concat(img_batch_real, img_batch_fake, dim=0)
                label_batch = mx.nd.concat(label_real, label_fake, dim=0)
                dis_batch = mx.io.DataBatch([img_batch], label=[label_batch])
                # forward and backward
                dis.forward(dis_batch)
                dis.backward()
                dis.update()
                lossD.append(dis.get_outputs()[0].asnumpy())
            out_D = dis.get_outputs()
            #  gradD = dis.get_input_grads()[0].asnumpy()
### added
        #  for img,label in it:
        #      if img.shape[0] != batch_size:
        #          continue
###
            ## train generator
            for gen_num in range(train_gen_iternums):
                noise_array = generator.gen_noise_uniform(noise_shape,(-1,1))
                noise_batch = mx.io.DataBatch([noise_array], label=None)
                # generator forward
                gen.forward(noise_batch)
                gen_output = gen.get_outputs()[0]
                label = mx.nd.ones((batch_size,1), ctx=mx.gpu())
                gen_batch = mx.io.DataBatch([gen_output], label=[label])
                dis.forward(gen_batch)
                dis.backward()
                grad = dis.get_input_grads()[0]
                gen.backward([grad])
                gen.update()
                lossG.append(dis.get_outputs()[0].asnumpy())
            out_G = dis.get_outputs()
            #  gradG = gen.get_input_grads()[0].asnumpy()
            #  print(out_G[1].shape)


            i += 1
            if i % 50 == 0:
                lossD_val = lossD[-1]
                lossG_val = lossG[-1]
                Dacc_real, Dacc_fake = dis_acc(out_D[1].asnumpy(), batch_size)
                #  print('gradG:')
                #  print(gradG)
                #  print('gradD:')
                #  print(gradD)
                print("epoch {}, iter {}, lossD {}, lossG {}, Dacc_real {}, Dacc_fake {}".format(e,i,np.round(lossD_val, 4), np.round(lossG_val, 4), Dacc_real, Dacc_fake))
                if draw and i % 200 == 0:
                    img_data = np.ceil((gen.get_outputs()[0].asnumpy()+1)*128).astype(np.uint8)
                    ImgHandler.draw(img_data)

        ## save params
        if e != 0 and e % save_epoch_num == 0:
            gen.save_checkpoint(save_path+'gen', 0, True)
            dis.save_checkpoint(save_path+'dis', 0, True)

    visualize.draw_loss([np.array(lossD),np.array(lossG)], ['lossD','lossG'], 2)




if __name__ == "__main__":
    train_gan(draw=True)



