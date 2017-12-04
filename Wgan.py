import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from ops import tlib,plot,save_images,mnist

flags = tf.app.flags

flags.DEFINE_integer('input_height',64,'input image height')
flags.DEFINE_integer("input_widht",64,'input image width')
flags.DEFINE_integer("batch_size",50,'input batch size')
flags.DEFINE_integer("input_channel",1,'input channel size')
flags.DEFINE_integer("out_height",64,'output height')
flags.DEFINE_integer("out_width",64,'output width')
flags.DEFINE_integer("z_dim",128,'generator input dim')
flags.DEFINE_integer("n_class",10,'number class')
flags.DEFINE_integer("DIM",64,'Model dimensionality')
flags.DEFINE_integer("Out_DIm",784,"output_dim")
flags.DEFINE_boolean("is_clip",False,"is clip")
flags.DEFINE_boolean("is_l2",False,"is L2")
flags.DEFINE_float("l2_regular",1,"l2 regularization")
flags.DEFINE_integer("iter_range",10000," iter range")
flags.DEFINE_integer("disc_inter",5,"disc iter")
flags.DEFINE_boolean("is_gp",True,"is gp")
flags.DEFINE_boolean("is_fsr",False,"is feasible set reduction")
flags.DEFINE_integer("lambda_reg",16,"is regularize fsr")
FLAGS = flags.FLAGS



def Generator(z,labels=None,reuse=False,nums=50):
    with tf.variable_scope("Generator") as scope:
        #z_labels = tf.concat([z,labels],1)
        if reuse:
            scope.reuse_variables()
        if z is None:
            z = tf.random_normal([nums,FLAGS.z_dim])
        if labels is not None:
            z = tf.concat([z,labels],1)
        oh,ow =flags.FLAGS.out_height,flags.FLAGS.out_width

        z_labels = tlib.fc(z,4*4*4*FLAGS.DIM,scope="project")

        out_put = tf.nn.relu(z_labels)

        out_put = tf.reshape(out_put,[-1,4,4,4*FLAGS.DIM])

        dconv1 = tlib.Con2D_transpose(out_put,[nums,8,8,2*FLAGS.DIM],5,2,scope="conv2D_transpose1")

        h1 = tf.nn.relu(dconv1)

        h1 = h1[:,:7,:7,:]

        dconv2=tlib.Con2D_transpose(h1,[nums,14,14,FLAGS.DIM],5,2,scope="conv2D_transpose2")

        h2= tf.nn.relu(dconv2)

        dconv3 = tlib.Con2D_transpose(h2,[nums,28,28,FLAGS.input_channel],5,2,scope="conv2D_transpose3")

        h3 = tf.nn.sigmoid(dconv3)

    return tf.reshape(h3,[-1,FLAGS.Out_DIm])

def Discriminator(input,reuse=False):
    with tf.variable_scope("Discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        image = tf.reshape(input,[-1,28,28,1])

        conv1 = tlib.Con2D(image,FLAGS.DIM,5,2,scope="conv1")

        relu1 = tlib.leaky_relu(conv1)

        conv2 = tlib.Con2D(relu1,2*FLAGS.DIM,5,2,scope="conv2")

        relu2 = tlib.leaky_relu(conv2)

        conv3= tlib.Con2D(relu2,4*FLAGS.DIM,5,2,scope="conv3")

        relu3 =tlib.leaky_relu(conv3)

        out_put = tf.reshape(relu3,[-1,4*4*4*FLAGS.DIM])

        fc1= tlib.fc(out_put,1,scope="fc1")

    return fc1

train_data,dev_data,test_data= mnist.load(FLAGS.batch_size,FLAGS.batch_size)

def inf_train_gen():
    while True:
        for images,targets in train_data():
            yield images


def main(_):
    X_image = tf.placeholder(tf.float32,[None,FLAGS.Out_DIm])
    #y_label = tf.placeholder(tf.float32,[None,FLAGS.n_class])

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

    gen_cost  = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(1e-3,global_step,200,0.96,staircase=True)

    clip_ops = []
    if FLAGS.is_clip :
        clip_bound=[-.01, .01]
        for v in disc_params:
            clip_ops.append(tf.assign(v,tf.clip_by_value(v,clip_bound[0],clip_bound[1])))

        clip_weight_clip = tf.group(*clip_ops)

    elif FLAGS.is_l2:
        for v in disc_params:
            tf.add_to_collection("loss",tf.multiply(tf.nn.l2_loss(v),FLAGS.l2_regular))
    elif FLAGS.is_gp:
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
        disc_cost += 10*gradient_penalty

    tf.add_to_collection("loss",disc_cost)
    dis_losses = tf.add_n(tf.get_collection_ref("loss"))

    #dis_losses = disc_cost
    gen_train = tf.train.AdamOptimizer(learning_rate, beta1=0.5,
        beta2=0.9).minimize(gen_cost,global_step=global_step,var_list=gen_params)
    disc_train = tf.train.AdamOptimizer(learning_rate,beta1=0.5,
        beta2=0.9).minimize(dis_losses,global_step=global_step,var_list=disc_params)

    #tensor_noise = tf.random_normal([128,128])
    tensor_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
    gen_save_image = Generator(tensor_noise,reuse=True,nums=128)

    #mnist_data  = input_data.read_data_sets("../data",one_hot=True)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        gen = inf_train_gen()
        for i in xrange(FLAGS.iter_range):
            start_time = time.time()
            #data = mnist_data.train.next_batch(FLAGS.batch_size)
            data = gen.next()
            if i >0:
                _genc,_ = sess.run([gen_cost,gen_train],feed_dict={X_image:data})

            for x in xrange(FLAGS.disc_inter):
                _disc,_ = sess.run([disc_cost,disc_train],feed_dict={X_image:data})
            if i>0:
                D_real,D_fake = sess.run([disc_real,disc_fake],feed_dict={X_image:data})
                #plot.plot("Generator_cost",_genc)
                plot.plot("Discriminator",_disc)
                plot.plot("D_real",np.mean(D_real))
                plot.plot("D_fake",np.mean(D_fake))
                plot.plot('time', time.time() - start_time)
            #if clip_ops is not None:
            #    sess.run(clip_weight_clip)
            if i%1000==999:
                print "***************************************************"
                print "D:"
                print D_real-D_fake
                print "****************************************************"
            if i%100==99:
                image = sess.run(gen_save_image)
                save_images.save_images(image.reshape((128,28,28)),"./gen_image_{}.png".format(i))
                val_dis_list=[]
                #for n in xrange(20):
                    #val_data= mnist_data.validation.next_batch(FLAGS.batch_size)

                    #_val_disc = sess.run(disc_cost,feed_dict={X_image:val_data[0],y_label:val_data[1]})
                #    val_dis_list.append(_val_disc)
                for images_,_ in dev_data():
                    _dev_disc_cost=sess.run(disc_cost,feed_dict={X_image:images_})
                    val_dis_list.append(_dev_disc_cost)
                plot.plot("val_cost",np.mean(val_dis_list))
            if i<5 or i%100==99:
                plot.flush()

            plot.tick()

if __name__ == '__main__':
   tf.app.run()
