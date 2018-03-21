#!/usr/bin/python


import mxnet as mx
import os
import core.visualize as visualize
import core.config as config



def test():
    save_path = os.getcwd() + config.save_path
    batch_size = config.batch_size

    assert os.path.exists(save_path), 'need train the model first'
    assert len(os.listdir(save_path)) != 0, 'need train the model first'


    modG = mx.mod.Module.load(save_path+'/modG', 0, True, context=mx.gpu(),
            data_names=['noise', 'con_label'], label_names=None)
    modG.bind(data_shapes=[('noise',(batch_size, 100, 1, 1)),
            ('con_label', (batch_size, 1, 1, 1))])

    noise = mx.nd.random.uniform(-1, 1, (batch_size, 100, 1, 1))
    con_label = mx.nd.ones(shape=(128,))
    for i in range(10):
        con_label[i*10:(i+1)*10] = i

    data_batch = mx.io.DataBatch(data=[noise, con_label])
    modG.forward(data_batch, is_train=False)
    gen_img = modG.get_outputs()[0].asnumpy()

    img_handler = visualize.ImgGrids(2)
    img_handler.draw(gen_img[:100])


if __name__ == '__main__':
    test()
    input('type in anything to stop the program')
