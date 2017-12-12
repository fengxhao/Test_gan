import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

from ops import mmd

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise) #input 50*128   128 * [4*4*4*64]  out 50* [4*4*4*64]
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])  # 50* [4*64, 4, 4]

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output) #50 * [8,8,64*2] -> 50*[64*2,8,8]
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]   #50*[64*2,7,7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output) # 50*[64*2,14,14]
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output) #50*[1,28,28]
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])  #50* [784]

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2) # 50 [64,14,14]
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2) # 50 [64,7,7]
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2) # 50 [128,4,4]
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM]) # 50* (4*4*4*64)
    output_1 = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output) #50*1
    out_logit = lib.ops.linear.Linear('Discriminator.logit',4*4*4*DIM,10,output)

    return output_1,out_logit #50
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_label = tf.placeholder(tf.int32,shape=[BATCH_SIZE])

fake_data = Generator(BATCH_SIZE)

disc_real,real_logit = Discriminator(real_data)
disc_fake,fake_logit= Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

alpha = tf.random_uniform(
   shape=[BATCH_SIZE,1],
   minval=0.,
   maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
inter_img,a=Discriminator(interpolates)
gradients = tf.gradients(inter_img, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
gp_cost= 10*gradient_penalty

class_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_label,logits=real_logit))
class_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_label,logits=fake_logit))

gen_cost  = -1*tf.reduce_mean(disc_fake)+class_loss_fake
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)+(class_loss_fake+class_loss_real)+gp_cost

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(disc_cost, var_list=disc_params)

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
fix_label_onehot = tf.one_hot(tf.reshape(fixed_labels,[100]),10)
fixed_noise_samples = Generator(100,noise=fixed_noise)

def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((100, 28, 28)),
        './con_cifar_image/samples_{}.jpg'.format(frame)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images,targets

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# Train loop
with tf.Session(config=config) as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        start_time = time.time()
        if iteration > 0:
            _ = session.run(gen_train_op,feed_dict={real_data:_data,real_label:_label})
        for i in xrange(CRITIC_ITERS):
            _data,_label = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op],feed_dict={real_data: _data,real_label:_label})
            d_real,d_fake=session.run([disc_real,disc_fake],feed_dict={real_data:_data,real_label:_label})
        if iteration>0:
            lib.plot.plot('train disc cost label', _disc_cost)
            lib.plot.plot('D_real label',np.mean(d_real))
            lib.plot.plot('D_fake label',np.mean(d_fake))
            lib.plot.plot('time', time.time() - start_time)

        #Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
           dev_disc_costs = []
           for images,i_label in dev_gen():
               _dev_disc_cost = session.run(
                   disc_cost,
                   feed_dict={real_data: images,real_label:i_label}
               )
               dev_disc_costs.append(_dev_disc_cost)
           lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
           generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()
        lib.plot.tick()
