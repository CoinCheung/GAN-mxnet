

discriminator_type = 'lenet5'
#  discriminator_type = 'deep_convolution'

# layer symbol parameters
bn_eps = 1e-5
leaky_slope = 0.2


# image parameters
batch_size = 64
img_channels = 1
img_shape = (batch_size, img_channels, 28, 28)
noise_shape = (batch_size,100,1,1)


# training control parameters
epoch = 10
if_save_params = True
save_each_epoch = 5
if_drawing = True
print_iter_num = 50
draw_iter_num = 50


# optimizer parameters
gen_optimizer = 'adam'
dis_optimizer = 'adam'
# normal gan
gen_optimizer_params = (('learning_rate', 2e-4), ('beta1',0.5), ('wd', 5e-5))
dis_optimizer_params = (('learning_rate',2e-4),('beta1', 0.5),('wd', 5e-5))
# dc gan
#  gen_optimizer_params = (('learning_rate', 2e-4), ('beta1',0.5), ('wd', 0))
#  dis_optimizer_params = (('learning_rate',2e-4),('beta1', 0.5),('wd', 0))


