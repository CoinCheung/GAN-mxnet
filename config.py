
'''
    when it works with usual discriminator and generator, a weigth_decay of 1e-4 works generally better
'''

GAN_type = 'usual_gan'

# layer symbol parameters
bn_eps = 1e-5
leaky_slope = 0.01


# image parameters
batch_size = 64
img_channels = 1
img_shape = (batch_size, img_channels, 28, 28)
noise_shape = (batch_size,100,1,1)


# training control parameters
is_test = False
train_gen_iternums = 1
train_dis_iternums = 1
epoch = 10
save_each_epoch = 5


# optimizer parameters
gen_optimizer = 'adam'
dis_optimizer = 'adam'
gen_optimizer_params = (('learning_rate', 2e-4), ('beta1',0.5), ('wd', 1e-4))
dis_optimizer_params = (('learning_rate',2e-4),('beta1', 0.5),('wd', 1e-4))



