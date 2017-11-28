import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from ops import tlib,plot,save_images,mnist,mmd,inception_score,cifar10

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images

flags = tf.app.flags

flags.DEFINE_integer('input_height',32,'input image height')
flags.DEFINE_integer("input_widht",32,'input image width')
flags.DEFINE_integer("batch_size",50,'input batch size')
flags.DEFINE_integer("input_channel",3,'input channel size')
flags.DEFINE_integer("out_height",32,'output height')
flags.DEFINE_integer("out_width",32,'output width')
flags.DEFINE_integer("z_dim",128,'generator input dim')
flags.DEFINE_integer("n_class",10,'number class')
flags.DEFINE_integer("DIM",128,'Model dimensionality')
flags.DEFINE_integer("Out_DIm",3072,"output_dim 32*32*3")
flags.DEFINE_integer("iter_range",10000," iter range")
flags.DEFINE_integer("disc_inter",5,"disc iter")
flags.DEFINE_boolean("is_gp",True,"is gp")
flags.DEFINE_boolean("is_fsr",False,"is feasible set reduction")
flags.DEFINE_integer("lambda_reg",16,"is regularize fsr")
FLAGS = flags.FLAGS

DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 50 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def Generator(n_samples=None,noise=None,labels=None,reuse=False,nums=50):
    with tf.variable_scope("Generator") as scope:
        if reuse:
            scope.reuse_variables()
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
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
def Discriminator(input,reuse=False):
    with tf.variable_scope("Discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        output = tf.reshape(input, [-1, 3, 32, 32])
        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
        output = LeakyReLU(output)
        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
        output = LeakyReLU(output)
        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.reshape(output, [-1, 4*4*4*DIM])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)
    return tf.reshape(output, [-1])

def Generator_k(z,labels=None,reuse=False,nums=50):
    with tf.variable_scope("Generator") as scope:
        if reuse:
            scope.reuse_variables()
        if z is None:
            z = tf.random_normal([nums,FLAGS.z_dim])
        if labels is not None:
            z = tf.concat([z,labels],1)
        z_labels = tlib.fc(z,4*4*4*FLAGS.DIM,scope="project")
        bn1 = tlib.bn(z_labels,scope="bn1")
        out_put = tf.nn.relu(bn1)
        out_put = tf.reshape(out_put,[-1,4,4,4*FLAGS.DIM])
        dconv1 = tlib.Con2D_transpose(out_put,[nums,8,8,2*FLAGS.DIM],5,2,scope="conv2D_transpose1")
        bnconv1 = tlib.bn(dconv1,scope="bn2")
        h1 = tf.nn.relu(bnconv1)
        dconv2=tlib.Con2D_transpose(h1,[nums,16,16,FLAGS.DIM],5,2,scope="conv2D_transpose2")
        bnconv2 =tlib.bn(dconv2,scope="bn3")
        h2= tf.nn.relu(bnconv2)
        dconv3 = tlib.Con2D_transpose(h2,[nums,32,32,FLAGS.input_channel],5,2,scope="conv2D_transpose3")
        h3 = tf.tanh(dconv3)
    return tf.reshape(h3,[-1,FLAGS.Out_DIm])

def Discriminator_k(input,reuse=False):
    with tf.variable_scope("Discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        image = tf.transpose(tf.reshape(input,[-1,3,32,32]),perm=[0,2,3,1])
        conv1 = tlib.Con2D(image,FLAGS.DIM,5,2,scope="conv1")
        relu1 = tlib.leaky_relu(conv1)
        conv2 = tlib.Con2D(relu1,2*FLAGS.DIM,5,2,scope="conv2")
        relu2 = tlib.leaky_relu(conv2)
        conv3= tlib.Con2D(relu2,4*FLAGS.DIM,5,2,scope="conv3")
        relu3 =tlib.leaky_relu(conv3)
        out_put = tf.reshape(relu3,[-1,4*4*4*FLAGS.DIM])
        fc1= tlib.fc(out_put,1,scope="fc1")
    return tf.reshape(fc1, [-1])

data_dir="/home/shen/fh/data/cifar10"
train_data,dev_data= cifar10.load(FLAGS.batch_size,data_dir)

def inf_train_gen():
    while True:
        for images,_ in train_data():
            yield images


def main(_):
    X_image_int = tf.placeholder(tf.int32,[FLAGS.batch_size,FLAGS.Out_DIm])
    X_image =2*((tf.cast(X_image_int, tf.float32)/255.)-.5)

    z=tf.random_normal([FLAGS.batch_size,FLAGS.z_dim])
    G_image = Generator_k(z)

    disc_real = Discriminator_k(X_image)
    disc_fake = Discriminator_k(G_image,reuse=True)

    gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
    disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator")


    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[FLAGS.batch_size,1],
        minval=0.,
        maxval=1.
    )
    differences = G_image - X_image
    interpolates = X_image + (alpha*differences)
    gradients = tf.gradients(Discriminator_k(interpolates,reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

    tensor_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
    gen_save_image = Generator_k(z=tensor_noise,reuse=True,nums=128)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        gen = inf_train_gen()
        for i in xrange(FLAGS.iter_range):
            start_time = time.time()
            data = gen.next()
            if i >0:
                _genc,_ = sess.run([gen_cost,gen_train],feed_dict={X_image_int:data})

            for x in xrange(FLAGS.disc_inter):
                _disc,_ = sess.run([disc_cost,disc_train],feed_dict={X_image_int:data})
            if i>0:
                D_real,D_fake = sess.run([disc_real,disc_fake],feed_dict={X_image_int:data})
                #plot.plot("Generator_cost",_genc)
                plot.plot("Discriminator",_disc)
                plot.plot("D_real",np.mean(D_real))
                plot.plot("D_fake",np.mean(D_fake))
                plot.plot('time', time.time() - start_time)

#*******************************save image***************************************
            if i%100==99:
                image = sess.run(gen_save_image)
                images_ = ((image+1.)*(255./2)).astype('int32')
                lib.save_images.save_images(images_.reshape((128, 32, 32, 3)), 'samples_{}.jpg'.format(i))

                val_dis_list=[]
                for images_,_ in dev_data():
                    _dev_disc_cost=sess.run(disc_cost,feed_dict={X_image_int:images_})
                    val_dis_list.append(_dev_disc_cost)
                plot.plot("val_cost",np.mean(val_dis_list))

            if i<5 or i%100==99:
                plot.flush()

            plot.tick()

if __name__ == '__main__':
   tf.app.run()
