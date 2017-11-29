import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from ops import tlib,plot,save_images,mnist,mmd,inception_score,cifar10

flags = tf.app.flags

flags.DEFINE_integer('input_height',32,'input image height')
flags.DEFINE_integer("input_widht",32,'input image width')
flags.DEFINE_integer("batch_size",49,'input batch size')
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
flags.DEFINE_string("log_dir","/home/shen/fh/log_wgan","log dir for wgan")
FLAGS = flags.FLAGS



def Generator(z,labels=None,reuse=False,nums=49):
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

        conv1 = tlib.Con2D(input,FLAGS.DIM,5,2,scope="conv1")

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
        for images,targets in train_data():
            yield images

def imageRearrange(image, block=8):
    image = tf.slice(image, [0, 0, 0, 0], [block * block, -1, -1, -1])
    x1 = tf.batch_to_space(image, [[0, 0], [0, 0]], block)
    image_r = tf.reshape(tf.transpose(tf.reshape(x1,
        [32, block, 32, block, 3])
        , [1, 0, 3, 2, 4]),
        [1, 32 * block, 32 * block, 3])
    return image_r

def main(_):
    X_image_int = tf.placeholder(tf.int32,[FLAGS.batch_size,FLAGS.Out_DIm])
    X_image =2*((tf.cast(X_image_int, tf.float32)/255.)-.5)#BCHW

    z=tf.random_normal([FLAGS.batch_size,FLAGS.z_dim])
    G_imge = Generator(z)

    real_img= tf.transpose(tf.reshape(X_image,[-1,3,32,32]),perm=[0,2,3,1])#BHWC

    real_img_rbg = tf.cast(tf.transpose(tf.reshape(X_image_int,[-1,3,32,32]),perm=[0,2,3,1]),tf.float32)

    fake_img= tf.reshape(G_imge,[-1,32,32,3])
    fake_img_rbg = tf.cast((fake_img+1.)*(255./2),tf.float32)

    tf.summary.image("train/input image",imageRearrange(real_img_rbg,7))
    tf.summary.image("train/gen image",imageRearrange(fake_img_rbg,7))

    disc_real = Discriminator(real_img)
    disc_fake = Discriminator(fake_img,True)

    #***reshpe real data n*3072
    X_image_trans=tf.reshape(real_img,[-1,FLAGS.Out_DIm])

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
        differences = X_image_trans - G_imge
        interpolates = X_image_trans + (alpha*differences)
        inter_image = tf.reshape(interpolates,[-1,32,32,3])
        gradients = tf.gradients(Discriminator(inter_image,reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        gp_cost= 10*gradient_penalty

    disc_cost+=gp_cost

    tf.summary.scalar("Generator_cost",gen_cost)
    tf.summary.scalar("Discriminator_cost",disc_cost)

    gen_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5,
        beta2=0.9).minimize(gen_cost,global_step=global_step,var_list=gen_params)
    disc_train = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5,
        beta2=0.9).minimize(disc_cost,global_step=global_step,var_list=disc_params)

    #tensor_noise = tf.random_normal([128,128])
    tensor_noise = tf.constant(np.random.normal(size=(64, 128)).astype('float32'))
    gen_save_image = Generator(tensor_noise,reuse=True,nums=64)
    generator_img = tf.reshape(gen_save_image,[-1,32,32,3])
    gen_img_rbg=tf.cast((generator_img+1.)*(255./2),tf.float32)

    tf.summary.image("Test/dev image",imageRearrange(gen_img_rbg))

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333) 
    config = tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth=True 	
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        train_summary = tf.summary.merge_all()
        write = tf.summary.FileWriter(logdir=FLAGS.log_dir,graph=sess.graph)
        gen = inf_train_gen()
        for i in xrange(FLAGS.iter_range):
            start_time = time.time()
            data = gen.next()

#*********************************inception score******************************************
            if i%100000==99999:
                all_samples = []
                gen_tensor_flow = tf.random_normal([64,128])
                gen_img = Generator(gen_tensor_flow,reuse=True,nums=64)
                for i in xrange(10):
                    all_samples.append(sess.run(gen_img))
                all_samples = np.concatenate(all_samples, axis=0)
                all_samples = ((all_samples+1.)*(255./2)).astype('int32')
                all_samples = all_samples.reshape((-1, 32, 32,3))
                score = inception_score.get_inception_score(list(all_samples))
                plot.plot("inception score:",score[0])

            if i >0:
                _genc,_ = sess.run([gen_cost,gen_train],feed_dict={X_image_int:data})

            for x in xrange(FLAGS.disc_inter):
                _disc,_,summary_str,step= sess.run([disc_cost,disc_train,train_summary,global_step],feed_dict={X_image_int:data})

            if i%10==0 and i>0:
                write.add_summary(summary_str,global_step=i)
            if i>-1:
                D_real,D_fake,gp_cost_ = sess.run([disc_real,disc_fake,gp_cost],feed_dict={X_image_int:data})
                plot.plot("Discriminator",_disc)
                plot.plot("D_real",np.mean(D_real))
                plot.plot("D_fake",np.mean(D_fake))
                plot.plot("gp_cost:",gp_cost_)
                plot.plot('time', time.time() - start_time)


            if i<5 or i%100==99:
                plot.flush()

            plot.tick()

if __name__ == '__main__':
   tf.app.run()
