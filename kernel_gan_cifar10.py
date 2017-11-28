import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from ops import tlib,plot,save_images,mnist,mmd,inception_score,cifar10

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



def Generator(z,labels=None,reuse=False,nums=50):
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

def Discriminator(input,reuse=False):
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





data_dir="/home/feng/ipyhthon/GAN_code/data/cifar-10"
train_data,dev_data= cifar10.load(FLAGS.batch_size,data_dir)

def inf_train_gen():
    while True:
        for images,targets in train_data():
            yield images


def main(_):
    X_image_int = tf.placeholder(tf.int32,[FLAGS.batch_size,FLAGS.Out_DIm])
    X_image =2*((tf.cast(X_image_int, tf.float32)/255.)-.5)

    z=tf.random_normal([FLAGS.batch_size,FLAGS.z_dim])
    G_image = Generator(z)

    disc_real = Discriminator(X_image)
    disc_fake = Discriminator(G_image,True)

    gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
    disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator")

    FSR_cost=0
    #********feasible set reduce***************
    if FLAGS.is_fsr:
        reduce_cost = tf.reduce_mean(disc_fake) -tf.reduce_mean(disc_real)
        FSR_cost = tf.nn.relu(reduce_cost)

    #******************************************
    bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
   #kernel_cost = mmd.mix_rbf_mmd2(disc_real,disc_fake,sigmas=bandwidths)

    # gen_cost  = kernel_cost
    # disc_cost = -1*kernel_cost
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(1e-3,global_step,200,0.96,staircase=True)

    gp_cost=0
    if FLAGS.is_gp:
        alpha = tf.random_uniform(
            shape=[FLAGS.batch_size,1],
            minval=0.,
            maxval=1.
        )
        differences = G_image - X_image
        interpolates = X_image + (alpha*differences)
        gradients = tf.gradients(Discriminator(interpolates,reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        gp_cost= 10*gradient_penalty

    disc_cost+=gp_cost

    gen_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,
        beta2=0.9).minimize(gen_cost,global_step=global_step,var_list=gen_params)
    disc_train = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,
        beta2=0.9).minimize(disc_cost,global_step=global_step,var_list=disc_params)

    #tensor_noise = tf.random_normal([128,128])
    tensor_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
    gen_save_image = Generator(tensor_noise,reuse=True,nums=128)

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
                D_real,D_fake,gp_cost_ = sess.run([disc_real,disc_fake,gp_cost],feed_dict={X_image_int:data})
                plot.plot("Discriminator",_disc)
                plot.plot("D_real",np.mean(D_real))
                plot.plot("D_fake",np.mean(D_fake))
                plot.plot("gp_cost:",gp_cost_)
                plot.plot('time', time.time() - start_time)
#*********************************inception score******************************************
            if i%100 ==99:
                all_samples = []
                gen_tensor_flow = tf.random_normal([128,128])
                gen_img = Generator(gen_tensor_flow,reuse=True,nums=128)
                for i in xrange(10):
                    all_samples.append(sess.run(gen_img))
                all_samples = np.concatenate(all_samples, axis=0)
                all_samples = ((all_samples+1.)*(255./2)).astype('int32')
                all_samples = all_samples.reshape((-1, 32, 32,3))
                score = inception_score.get_inception_score(list(all_samples))
                plot.plot("inception score:",score[0])
#*******************************save image***************************************
            if i%100==99:
                image = sess.run(gen_save_image)
                images_ = ((image+1.)*(255./2)).astype('int32')
                save_images.save_images(images_.reshape((128,32,32,3)),"./save_cifar_image/gen_image_{}.png".format(i))
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
