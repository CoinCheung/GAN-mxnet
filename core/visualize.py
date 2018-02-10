#!/usr/bin/python



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab



class ImgGrids(object):
    '''
    TODO: add annotations
    '''
    def __init__(self, fig_num):
        self.fig_num = fig_num
        self.fig = plt.figure(fig_num)
        plt.ion()

    def draw(self, img):
        batch_size, c, w, h = img.shape
        sqrtn = int(np.ceil(np.sqrt(batch_size)))

        plt.figure(self.fig_num)
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
        plt.show()
        plt.pause(0.001)





def draw_loss(loss, titles, fig_num):
    subnum = len(loss)

    _, ax = plt.subplots(subnum,1, num=fig_num, sharex=False,sharey=False)
    plt.ioff()
    for sn in range(subnum):
        x = np.arange(loss[sn].shape[0])
        y = loss[sn]
        ax[sn].plot(x,y)
        ax[sn].set_title(titles[sn])
    plt.show()


def draw_acc(acc, titles, fig_num):
    pass



if __name__ == "__main__":
    #  arr = np.random.randint(0,255,(32,32,4),dtype=np.uint8)
    arr = np.random.rand(32,32)
    #  fig = plt.figure()
    #  plt.imshow(arr)
    #  pylab.show()
    pass

    y1 = np.ones((10,))
    y2 = np.ones((12,))

    #  draw_loss([y1, y2])

    #  x = np.arange(100)
    #  y = np.sin(x)
    #
    #  # 表示2x2的子图布局，上下x坐标对齐，一下y坐标对齐
    #  _, ax = plt.subplots(2,2, sharex=True, sharey=True)
    #  ax[0][0].plot(x,y)
    #  ax[0][0].set_title("ddd")
    #  ax[1][0].plot(x,y)
    #  ax[1][0].set_title("ddd")
    #  ax[0][1].plot(x,y)
    #  ax[0][1].set_title("ddd")
    #  ax[1][1].plot(x,y)
    #  ax[1][1].set_title("ddd")
    #  plt.show()


    imhd = ImgGrids(3)
    for i in range(15):
        img = np.random.rand(16,1,28,28)
        imhd.draw(img)
        show_image(img, 1)


