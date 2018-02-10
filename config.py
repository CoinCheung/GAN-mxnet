


bn_eps = 1e-5
leaky_slope = 0.01

is_test = False

batch_size = 64
#  batch_size = 16
noise_shape = (batch_size, 96)
img_channel = 1
img_size = 28

train_gen_iternums = 1
train_dis_iternums = 1
epoch = 4


gen_optimizer = 'adam'
dis_optimizer = 'adam'
gen_learning_rate = 1e-3
dis_learning_rate = 1e-3
dis_weight_decay = 0.0006
gen_weight_decay = 0.0006

save_each_epoch = 5


