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

def Generator(n_samples,label=None, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    if label is not None:
        noise = tf.concat([noise,label],1)
    output = lib.ops.linear.Linear('Generator.Input', 138, 4*4*4*DIM, noise) #input 50*128   128 * [4*4*4*64]  out 50* [4*4*4*64]
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

    n_class = tf.nn.softmax(out_logit)
    return output_1,out_logit,n_class   #50
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_label = tf.placeholder(tf.int32,shape=[BATCH_SIZE])

real_label_onehot = tf.one_hot(real_label,10)

fake_data = Generator(BATCH_SIZE,real_label_onehot)

disc_real,disc_real_logit,soft_class_real = Discriminator(real_data)
disc_fake,disc_fake_logit,soft_class_fake = Discriminator(fake_data)

class_real = tf.argmax(soft_class_real,1)
class_fake = tf.argmax(soft_class_fake,1)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

class_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_label_onehot,logits=disc_real_logit))
class_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_label_onehot,logits=disc_fake_logit))

bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
kernel_cost = mmd.mix_rbf_mmd2(disc_real,disc_fake,sigmas=bandwidths,id=BATCH_SIZE)
ind_t=tf.placeholder(tf.int32,[11])
con_kernel_cost=0
for i in range(10):
    find_index = tf.where(tf.equal(real_label,i))
    Image_c = tf.gather(disc_real,find_index)
    Gimage_c = tf.gather(disc_fake,find_index)
    Image_c_s = tf.reshape(Image_c,[-1,1])
    Gimage_c_s = tf.reshape(Gimage_c,[-1,1])
    con_kernel_cost+=mmd.mix_rbf_mmd2(Image_c_s,Gimage_c_s,sigmas=bandwidths,id=ind_t[i])

alpha = tf.random_uniform(
   shape=[BATCH_SIZE,1],
   minval=0.,
   maxval=1.
)
reduce_cost = tf.reduce_mean(disc_fake) -tf.reduce_mean(disc_real)
FSR_cost = tf.nn.relu(reduce_cost)

differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
inter_img,a,b=Discriminator(interpolates)
gradients = tf.gradients(inter_img, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
gp_cost= 10*gradient_penalty

gen_cost  = kernel_cost+con_kernel_cost+(class_loss_fake) +10*FSR_cost
disc_cost = -1*(kernel_cost+con_kernel_cost)+(class_loss_real+class_loss_fake)+gp_cost+10*FSR_cost

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(disc_cost, var_list=disc_params)

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
fix_label_onehot = tf.one_hot(tf.reshape(fixed_labels,[100]),10)
fixed_noise_samples = Generator(100, label=fix_label_onehot,noise=fixed_noise)
#_,_,class_gen_label = Discriminator(fixed_noise_samples)
#gen_label = tf.argmax(class_gen_label,1)

def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((100, 28, 28)),
        './out1/samples_{}.jpg'.format(frame)
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
        #     continue
        if iteration > 0:
            _ = session.run(gen_train_op,feed_dict={real_data:_data,real_label:_label,ind_t:np.array(num_index)})
        for i in xrange(CRITIC_ITERS):
            _data,_label = gen.next()
            num_index=[]
            for ind in range(10):
                whlen = len(np.where(_label==ind)[0])
                if whlen==0:
                    whlen=1
                num_index.append(whlen)
            num = np.shape(np.unique(_label))[0]
            num_index.append(num)
            _disc_cost, _ = session.run([disc_cost, disc_train_op],feed_dict={real_data: _data,real_label:_label,ind_t:np.array(num_index)})
            d_real,d_fake,_con_kernel,real,fake,fsr,gp=session.run([disc_real,disc_fake,con_kernel_cost,class_loss_real,class_loss_fake,FSR_cost,gp_cost],feed_dict={real_data:_data,real_label:_label,ind_t:np.array(num_index)})
        if iteration>0:
            lib.plot.plot('train disc cost fsr', _disc_cost)
            lib.plot.plot('D_real fsr',np.mean(d_real))
            lib.plot.plot('D_fake fsr',np.mean(d_fake))
            lib.plot.plot('con_kernel_loss fsr',_con_kernel)
            lib.plot.plot('gp_cost fsr',gp)
            lib.plot.plot('fsr',fsr)
        if iteration%100==99:
            print "total_kernel_loss:"
            print session.run(kernel_cost,feed_dict={real_data:_data,real_label:_label,ind_t:np.array(num_index)})
            print "con_kernel_loss:"
            print _con_kernel
            print "real_class:"
            print real
            print "fake_class:"
            print fake
            lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        #if iteration % 100 == 99:
        #    dev_disc_costs = []
        #    for images,_ in dev_gen():
        #        _dev_disc_cost = session.run(
        #            disc_cost, 
        #            feed_dict={real_data: images}
        #        )
        #        dev_disc_costs.append(_dev_disc_cost)
        #    lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
