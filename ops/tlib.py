import tensorflow as tf

def Con2D(X,num_filter,kernel_size,stride_size,scope ="conv"):

    with tf.variable_scope(scope or "conv"):

        Weight = tf.get_variable('weight',[kernel_size,kernel_size,X.get_shape()[-1],num_filter],initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(X,Weight,[1,stride_size,stride_size,1],padding="SAME")
        bias = tf.get_variable("bias",[num_filter],initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv,bias)
    return conv

def Con2D_transpose(X,out_shape,kernel_size,stride,scope="conv_transpose"):

    with tf.variable_scope(scope or "conv_transpose"):
        Weight = tf.get_variable('weight',[kernel_size,kernel_size,out_shape[-1],X.get_shape()[-1]],initializer=tf.truncated_normal_initializer(stddev=0.02))
        con_transpose=tf.nn.conv2d_transpose(X,Weight,out_shape,[1,stride,stride,1])
        bias = tf.get_variable('baise',[out_shape[-1]],initializer=tf.constant_initializer(0.0))
        con_transpose = tf.nn.bias_add(con_transpose,bias)
    return con_transpose

def fc(X,num_out,scope="fc"):

    with tf.variable_scope(scope or "fc"):
        Weight = tf.get_variable('weight',[X.get_shape()[-1],num_out],initializer=tf.truncated_normal_initializer(stddev=0.02))
        fc= tf.matmul(X,Weight)
        bias = tf.get_variable("bias",[num_out],initializer=tf.constant_initializer(0.0))
        #bias = tf.Variable(tf.constant(0,[num_out]))
        fc = tf.nn.bias_add(fc,bias)
    return fc

def bn(X,decay=0.95,epsilon =1e-5,scale=True,reuse = True,scope="batch_normal"):
    bn =  tf.layers.batch_normalization(X,momentum=decay,epsilon=epsilon,scale=scale,training=True,reuse=reuse,name=scope)
    return bn

def leaky_relu(X,leak=0.2):
    return tf.maximum(X,X*leak)
