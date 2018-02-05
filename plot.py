#!/usr/bin/python



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab



def show_image(img):
    '''
    visulize the images given by the data batch
    params:
        img: a numpy array with shape(batch_size, weigth, height, channels)
    '''
    batch_size, c, w, h = img.shape
    sqrtn = int(np.ceil(np.sqrt(batch_size)))

    fig = plt.figure(figsize=(2,2))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i,image in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        if c == 3 or c == 4:
            plt.imshow(image)
        else:
            plt.imshow(image.reshape(w,h))
    pylab.show()



if __name__ == "__main__":
    #  arr = np.random.randint(0,255,(32,32,4),dtype=np.uint8)
    arr = np.random.rand(32,32)
    fig = plt.figure()
    plt.imshow(arr)
    pylab.show()
    pass
