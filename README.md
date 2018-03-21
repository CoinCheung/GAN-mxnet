# GAN-mxnet

This is a mxnet implementation of GAN for the purpose of generating MNIST dataset pictures.

# Original GAN structure


### About the model
Two sort of network structures are supported: Lenet-5 and deep convolution.

1. discriminators  
The discriminator structure of Lenet-5 based model is as follows:

* **Convolutional layer**: 32 filters of 5x5, stride 1, pad 0, 0.2 slope leaky relu 
* **Max pooling layer**: window size 2x2, stride 2
* **Convolutional layer**: 64 filters of 5x5, stride 1, pad 0, 0.2 slope leaky relu
* **Max pooling layer**: window size 2x2, stride 2
* **Convolutional layer**: 128 filters of 5x5, stride 1, pad 0, 0.2 slope leaky relu
* **Max pooling layer**:  window size 2x2, stride 2, Flatten
* **Dense layer**: 1024 hidden nodes, 0.2 slope leaky relu
* **Dense layer**: 1 hidden node

The deep convolution network based model has the discriminator structure as follows:

* **Convolutional layer**: 128 filters of 4x4, stride 2, pad 1, 0.2 slope leaky relu
* **Convolutional layer**: 256 filters of 4x4, stride 2, pad 1, BatchNorm, 0.2 slope leaky relu
* **Convolutional layer**: 512 filters of 4x4, stride 2, pad 1, BatchNorm, 0.2 slope leaky relu
* **Convolutional layer**: 1024 filters of 4x4, stride 2, pad 1, BatchNorm, 0.2 slope leaky relu
* **Convolutional layer**: 1 filters of 4x4, stride 1, pad 0

2. generators  
As for the generators, they always have the structures opposite to their discriminator counterpart.  

3. input and output layers   
For the ease of computation and generalizing to other datasets with different data shapes, the input data will be first resized to 64x64 before fed to the model. In terms of the output layers, those of the two discriminators are both sigmoid cross entropy layers.


### Training
1. A bit configuration  
* Choosing models  
If Lenet-5 based models are to be used, one should open the file core/config.py and uncomment the line with 'lenet5' and comment the line with 'deep_convolution':
```python
    discriminator_type = 'lenet5'
    # discriminator_type = 'deep_convolution'

```
If deep convolution network models are to be used, one will need to carry out the otherwise comment behavior.

* Draw the generated pictures  
By assigning the variable 'if_drawing' in the file core/config.py to be 'True', one could see the generated pictures plotted every several iterations. The period of plotting can be assigned by the variable 'draw_iter_num'.

* Save models periodically  
By default, the models will be saved to the directory of model_export each 5 epoches. One could switch off the saving function by assigning the variable if_save_params to be False in the file core/config.py.

There are also other behaviors supported, one may swich on/off them in the file core/config.py.


2. Train the network  
Start training by running the python script train.py in the project directory:
```sh
    $ cd GAN-mxnet
    $ python3 train.py
```
In general, the Lenet-5 based models will train faster than the deep convolution based models. 

After the process of training, the losses of the generator and discriminator will be plotted which may indicate the state of training.



## Conditional GAN structure
A conditional GAN(CGAN) is also provided to generate MNIST pictures. One obvious advantage of CGANs is that they allow the user to manipulate the behavior of the model. In this project, by assigning the condition to be the number classes to which the pictures belong to, we could control which numbers the generator generates in the pictures.


### Training
Simply go the root directory of the CGAN, and run the training script:
```sh
    $ cd GAN-mxnet/CGAN
    $ python3 train.py
```
After training for less than an hour on gpus, the models should converge. The trained models will be exported to the directory of ```GAN-mxnet/CGAN/model-export```.


### Test
In the same directory, run the ```test.py``` script:
```sh
    $ python3 test.py
```
One will see that the model generates the picutures as assigned instead of randomly.




## Tips:
Here are the things I did for fine tuning which makes the generated pictures look better:
1. **No weight decay**, weight decay works the same as L2 regularization. It seems to work better in the classification tasks, but for the GAN training tasks.
2. **No leaky relu for generator**, leaky relu activation can be used in discriminators but they work not well for generators. So for generators, it is better to simply use plain relu activations.
3. **noise ranges**, It turns out that the generator with input noises drawn from the space [0, 1] works not as well as that with noises picked from the symmetric ragion [-1, 1]. 



I might have the wrong skills of training a GAN. If there are errors in these tips, please let me know and I will changes them as well as my comprehensions.
