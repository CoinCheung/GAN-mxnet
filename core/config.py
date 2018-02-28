
'''
    when it works with lenet5 discriminator and generator, a weigth_decay of 1e-4 works generally better
'''

#  discriminator_type = 'lenet5'
discriminator_type = 'deep_convolution'

# layer symbol parameters
bn_eps = 1e-5 + 1e-12
leaky_slope = 0.2


# image parameters
batch_size = 64
img_channels = 1
img_shape = (batch_size, img_channels, 28, 28)
noise_shape = (batch_size,100,1,1)


# training control parameters
epoch = 10
save_each_epoch = 5


# optimizer parameters
gen_optimizer = 'adam'
dis_optimizer = 'adam'
# normal gan
gen_optimizer_params = (('learning_rate', 2e-4), ('beta1',0.5), ('wd', 1e-4))
dis_optimizer_params = (('learning_rate',2e-4),('beta1', 0.5),('wd', 1e-4))
#  # dc gan
#  gen_optimizer_params = (('learning_rate', 2e-4), ('beta1',0.5), ('wd', 0))
#  dis_optimizer_params = (('learning_rate',2e-4),('beta1', 0.5),('wd', 0))


