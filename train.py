#!/usr/bin/python


import mxnet as mx
import numpy as np
import discriminator
import generator
import loss
import IO
import config
import plot



def train_gan():
    # control params
    train_gen_iternums = config.train_gen_iternums
    train_dis_iternums = config.train_dis_iternums
    eps = config.bn_eps
    test = config.is_test
    batch_size = config.batch_size
    img_size = config.img_size


    # syms
    real_batch = mx.sym.var('real_batch')
    fake_batch = mx.sym.var('fake_batch')
    noise = mx.sym.var("noise")
    gen_batch = generator.generator_conv(noise, batch_size, test, eps)

    loss_D, loss_G = loss.gan_loss(real_batch, fake_batch, batch_size)


    # modules
    # generator
    gen = mx.mod.Module(gen_batch, context=mx.gpu(), data_names=['noise'], label_names=None)
    gen.bind(data_shapes=[('noise',(batch_size, 1024))], label_shapes=None)
    gen.init_params()
    gen.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', 1e-1), ('beta1',0.5)))

    # loss used to train discriminator
    lossD_mod = mx.mod.Module(loss_D, context=mx.gpu(), data_names=['real_batch', 'fake_batch'], label_names=None)
    lossD_mod.bind(data_shapes=[('real_batch',(batch_size,1,img_size,img_size)), ('fake_batch', (batch_size,1,img_size,img_size))], label_shapes=None)
    lossD_mod.init_params()
    lossD_mod.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', 1e-1), ('beta1',0.5)))

     # loss used to train generator
    lossG_mod = mx.mod.Module(loss_G, context=mx.gpu(), data_names=['fake_batch'], label_names=None)
    lossG_mod.bind(data_shapes=[('fake_batch',(batch_size,1,img_size,img_size))], label_shapes=None, inputs_need_grad=True)
    lossG_mod.init_params()
    lossG_mod.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', 1e-3), ('beta1',0.5)))



    # training procedure
    it = IO.get_mnist_iter()
    epoch = 10
    i = 0
    for e in range(epoch):
        for img,label in it:
            ## train generator
            noise_array = generator.gen_noise_uniform((batch_size,1024),(-1,1))
            noise_batch = mx.io.DataBatch([noise_array], label=None)
            # generator forward
            gen.forward(noise_batch)
            gen_output = gen.get_outputs()[0]
            gen_loss_batch = mx.io.DataBatch([gen_output], label=None)
            lossG_mod.forward(gen_loss_batch)
            # generator backward
            lossG_mod.backward()
            gen_loss_grad = lossG_mod.get_input_grads()
            gen.backward(gen_loss_grad)
            gen.update()

            ## train discriminator
            noise_array = generator.gen_noise_uniform((batch_size,1024),(-1,1))
            noise_batch = mx.io.DataBatch([noise_array], label=None)
            gen.forward(noise_batch)
            fake_loss_batch = gen.get_outputs()[0]
            lossD_batch = mx.io.DataBatch([img.as_in_context(mx.gpu()),fake_loss_batch])
            # forward
            lossD_mod.forward(lossD_batch)
            lossD_mod.backward()
            lossD_mod.update()


            i += 1
            if i % 50 == 0:
                lossG_val = lossG_mod.get_outputs()[0].asnumpy()
                lossD_val = lossD_mod.get_outputs()[0].asnumpy()
                print("epoch {}, iter {}, lossD {}, lossG {}".format(e,i,lossD_val, lossG_val))
                #  print(gen_output.asnumpy())
                #  img_data = np.ceil((gen_output.asnumpy()+1)*128).astype(np.uint8)
                #  print(img_data)
                #  plot.show_image(img_data)



if __name__ == "__main__":
    train_gan()



