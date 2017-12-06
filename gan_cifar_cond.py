import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot
from ops import mmd
# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = '/home/feng/ipyhthon/GAN_code/data/cifar-10'
#DATA_DIR = '/home/shen/fh/data/cifar10'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 100000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator(n_samples,label, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    noisez = tf.concat([noise,label],1)
    output = lib.ops.linear.Linear('Generator.Input', 138, 4*4*4*DIM, noisez)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    fc1 = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    fc2= lib.ops.linear.Linear('Discriminator.Output1',4*4*4*DIM,10,output)

    return tf.reshape(fc1, [BATCH_SIZE,1]),fc2

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
real_label = tf.placeholder(tf.int32,shape=[BATCH_SIZE])
label_onehot =tf.one_hot(real_label,10)

fake_data = Generator(BATCH_SIZE,label_onehot)

disc_real,real_logit = Discriminator(real_data)
disc_fake,fake_logit = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

class_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_label,logits=real_logit))
class_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_label,logits=fake_logit))

#******************************************
bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
kernel_cost = mmd.mix_rbf_mmd2(disc_real,disc_fake,sigmas=bandwidths)

ind_t=tf.placeholder(tf.int32,[11])
cum = tf.placeholder(tf.float32)
con_kernel_cost=0
for i in range(10):
    find_index = tf.where(tf.equal(real_label,i))
    Image_c = tf.gather(disc_real,find_index)
    Gimage_c = tf.gather(disc_fake,find_index)
    Image_c_s = tf.reshape(Image_c,[-1,1])
    Gimage_c_s = tf.reshape(Gimage_c,[-1,1])
    con_kernel_cost+=mmd.mix_rbf_mmd2(Image_c_s,Gimage_c_s,sigmas=bandwidths,id=ind_t[i])


# gen_cost  = kernel_cost
# disc_cost = -1*kernel_cost

# gen_cost = -tf.reduce_mean(disc_fake)
# disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1],
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
inter_img,a =Discriminator(interpolates)
gradients = tf.gradients(inter_img, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
gp_cost = LAMBDA*gradient_penalty

con_kernel_cost =tf.div(con_kernel_cost,cum)
gen_cost  = con_kernel_cost+(class_loss_fake)
disc_cost = -1*(con_kernel_cost)+(class_loss_real)+gp_cost

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# For generating samples
fixed_noise_128 = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))

fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
fix_label_onehot = tf.one_hot(tf.reshape(fixed_labels,[100]),10)
fixed_noise_samples = Generator(100, label=fix_label_onehot,noise=fixed_noise_128)

def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), './save_cifar_image/samples_{}.jpg'.format(frame))

# For calculating inception score
fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
fake_onhot  = tf.one_hot(fake_labels_100,10)
samples_100 = Generator(100, fake_onhot)
def get_inception_score():
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterators
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images,labels in train_gen():
            yield images,labels

# Train loop
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op, feed_dict={real_data_int: _data,real_label:_label,ind_t:np.array(num_index),cum:num_index[10]})
        # Train critic
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
            _disc_cost,_,_gp,_con_cost,real,fake = session.run([disc_cost, disc_train_op,gp_cost,con_kernel_cost,class_loss_real,class_loss_fake], feed_dict={real_data_int: _data,real_label:_label,ind_t:np.array(num_index),cum:num_index[10]})

        #d_real,d_fake=session.run([disc_real,disc_fake],feed_dict={real_data_int:_data,real_label:_label,ind_t:np.array(num_index)})
        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('class_real',real)
        lib.plot.plot('class_fake',fake)
        lib.plot.plot('gp_cost',_gp)
        lib.plot.plot('con_kernel_cost',_con_cost)
        lib.plot.plot('time', time.time() - start_time)


        # Calculate inception score every 1K iters
        if iteration % 1000 == 999:
            inception_score = get_inception_score()
            lib.plot.plot('inception score', inception_score[0])

        # Calculate dev loss and generate samples every 100 iters
        # if iteration % 100 == 99:
        #     dev_disc_costs = []
        #     for images,de_label in dev_gen():
        #         num_index_de=[]
        #         for ind in range(10):
        #             whlen = len(np.where(de_label==ind)[0])
        #             if whlen==0:
        #                 whlen=1
        #             num_index_de.append(whlen)
        #         num = np.shape(np.unique(de_label))[0]
        #         num_index_de.append(num)
        #         _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images,_label:de_label,ind_t:np.array(num_index_de),cum:num_index_de[10]})
        #         dev_disc_costs.append(_dev_disc_cost)
        #     lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
