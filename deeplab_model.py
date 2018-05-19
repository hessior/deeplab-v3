import tensorflow as tf
import numpy as np


H = 160
W = 256

def deeplab_v3(inpu,  name, is_training=None):
    '''down size by 16'''
    with tf.variable_scope(name):
        conv1 = conv(inpu, 7,1,64,2,'conv1')
        bn1 = bn(conv1, is_training, name='bn1')
        relu1 = tf.nn.relu(conv1,'relu1')
        pool1 = tf.nn.max_pool(relu1, [1,3,3,1],[1,2,2,1],padding="SAME")

        res1 = res(pool1, 64, 256, 1, is_training, 'bk1res1')
        res2 = res(res1, 256, 256, 1, is_training,'bk1res2')
        res3 = res(res2, 256, 256, 1, is_training,'bk1res3')
        res4 = res(res3, 256, 512, 2, is_training,'bk2res1')
        res5 = res(res4, 512, 512, 1, is_training,'bk2res2')
        res6 = res(res5, 512, 512, 1, is_training,'bk2res3')
        res7 = res(res6, 512, 512, 1, is_training,'bk2res4')
        res8 = res(res7, 512, 1024,2, is_training,'bk3res1')
        res9 = res(res8, 1024,1024,1, is_training,'bk3res2')
        res10 = res(res9, 1024,1024,1, is_training,'bk3res3')
        res11 = res(res10, 1024,1024,1,is_training, 'bk3res4')
        res12 = res(res11, 1024,1024,1, is_training,'bk3res5')
        res13 = res(res12, 1024,1024,1, is_training,'bk3res6')
        
        res14 = res(res13, 1024,2048,1,is_training, 'bk4res1', atrous=2)
        res15 = res(res14, 2048,2048,1,is_training, 'bk4res2')
        res16 = res(res15, 2048,2048,1,is_training, 'bk4res3')
        x0 = conv(res16, 1, 2048, 2048, 1, 'pyrmd0')
        x1 = conv(res16, 3, 2048, 2048, 1, 'pyrmd1',atrous=3)
        x2 = conv(res16, 3, 2048, 2048, 1, 'pyrmd2',atrous=6)
        xb = tf.reduce_mean(res16, axis=[1,2], keepdims=True)

        x = tf.concat([res16, x0, x1, x2], axis=-1)
        logits = conv(x, 1, 2048*4, 4, 1, 'logits')
        logits = tf.image.resize_bilinear(logits, [H,W])
        pred = tf.nn.softmax(logits, axis=-1)

        #x = res16 + x0 + x1 + x2 +xb
        #logits = conv(x, 1, 2048, 1, 1, 'logits')
        #pred = tf.tanh(logits)
        
    return logits, pred



def conv(inpu, ks, in_chnl, out_chnl, stride, name, atrous=0):
    '''only padding=same'''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filt = tf.get_variable("filter",shape=[ks,ks,in_chnl,out_chnl],\
                               initializer=tf.truncated_normal_initializer(
                                   stddev=0.01))
        if atrous == 0:
            out = tf.nn.conv2d(inpu,filt,[1,stride,stride,1],padding="SAME")
        else:
            out = tf.nn.atrous_conv2d(inpu, filt, rate=atrous,padding="SAME")
    return out

def res(inpu, in_chnl, out_chnl, stride, is_training, name, atrous=0):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1 = conv(inpu, 1, in_chnl, out_chnl//4, stride, 'conv1', atrous)
        bn1 = bn(conv1, is_training, name='bn1')
        relu1 = tf.nn.relu(conv1)
        conv2 = conv(conv1, 3, out_chnl//4, out_chnl//4, 1, 'conv2', atrous)
        bn2 = bn(conv2, is_training, name='bn2')
        relu2 = tf.nn.relu(conv2)
        conv3 = conv(conv2, 1, out_chnl//4, out_chnl, 1, 'conv3', atrous)
        if in_chnl == out_chnl and stride==1:
            out = tf.nn.relu(bn(inpu + conv3, is_training, name='bn3'))
        else:
            inpu_chg = conv(inpu, 1, in_chnl, out_chnl, stride, 'inpu_change')
            out = tf.nn.relu(bn(inpu_chg + conv3, is_training, name='bn3'))
    return out

#############  this bn is from others, seems wrong
##def bn_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
##    with tf.variable_scope(scope,reuse=reuse):
##        shape = x.get_shape().as_list()
##        gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.constant_initializer(1.0), trainable=True)
##        beta = tf.get_variable("beta",[shape[-1]],initializer=tf.constant_initializer(0.0), trainable=True)
##        
##        moving_avg = tf.get_variable("moving_avg", [shape[-1]], initializer=tf.constant_initializer(0.0), trainable=False)
##        moving_var = tf.get_variable("moving_var", [shape[-1]], initializer=tf.constant_initializer(1.0), trainable=False)
##
##        if is_training:
##            # tf.nn.moments == Calculate the mean and the variance of the tensor x
##            avg, var = tf.nn.moments(x, np.arange(len(shape)-1), keep_dims=True)
##            avg=tf.reshape(avg, [avg.shape.as_list()[-1]])
##            var=tf.reshape(var, [var.shape.as_list()[-1]])
##            #update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
##            update_moving_avg=tf.assign(moving_avg, moving_avg*decay+avg*(1-decay))
##            #update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
##            update_moving_var=tf.assign(moving_var, moving_var*decay+var*(1-decay))
##            control_inputs = [update_moving_avg, update_moving_var]
##        else:
##            avg = moving_avg
##            var = moving_var
##            control_inputs = []
##        with tf.control_dependencies(control_inputs):
##            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
##
##    return output
##
##def bn(x, scope, is_training, epsilon=0.001, decay=0.99):
##    """
##    Returns a batch normalization layer that automatically switch between train and test phases based on the 
##    tensor is_training
##
##    Args:
##        x: input tensor
##        scope: scope name
##        is_training: boolean tensor or variable
##        epsilon: epsilon parameter - see batch_norm_layer
##        decay: epsilon parameter - see batch_norm_layer
##
##    Returns:
##        The correct batch normalization layer based on the value of is_training
##    """
##    #assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool
##
##    return tf.cond(
##        is_training,
##        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
##        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
##    )

##def _bn(inpu, is_training, decay=0.99, epsilon=0.001, name='bn_layer'):
##    batch_shape = inpu.get_shape().as_list()
##    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
##        beta = tf.get_variable("beta", initializer=\
##                                      tf.constant_initializer(0.0),\
##                                      shape=batch_shape[1:],\
##                                      trainable=True)
##        gamma = tf.get_variable("gamma", initializer=\
##                                      tf.constant_initializer(1.0),\
##                                      shape=batch_shape[1:],\
##                                      trainable=True)
##        moving_mean = tf.get_variable("moving_mean", initializer=\
##                                      tf.constant_initializer(0.0),\
##                                      shape=batch_shape[1:],\
##                                      trainable=False)
##        moving_var = tf.get_variable("moving_variance", initializer=\
##                                     tf.constant_initializer(1.0),\
##                                     shape=batch_shape[1:],\
##                                     trainable=False)
##        if is_training:
##            batch_mean, batch_var = tf.nn.moments(inpu, axes=[0])
##            update_mov_mean = tf.assign(moving_mean, moving_mean*decay+\
##                                        batch_mean*(1-decay))
##            update_mov_var = tf.assign(moving_var, moving_var*decay+\
##                                       batch_var*(1-decay))
##            controls = [update_mov_mean, update_mov_var]
##            with tf.control_dependencies(controls):
##                output = tf.nn.batch_normalization(inpu, batch_mean, batch_var,\
##                                                   offset=beta, scale=gamma,\
##                                                   variance_epsilon=epsilon,\
##                                                   name="bn")
##        else:
##            output = tf.nn.batch_normalization(inpu, moving_mean, moving_var,\
##                                               offset=beta, scale=gamma,\
##                                               variance_epsilon=epsilon,\
##                                               name="bn")
##    return output
##
##def bn(inpu, is_training, epsilon=0.001, decay=0.99, name="bn_layer"):
##    return tf.cond(is_training, lambda: _bn(inpu, True, epsilon, decay, name),\
##                                lambda: _bn(inpu, False, epsilon, decay, name))

def bn(inpu, is_training, epsilon=0.001, decay=0.99, name="bn_layer"):
    with tf.variable_scope(name):
        output = tf.contrib.layers.batch_norm(inpu, scale=True, is_training=\
                                              is_training, decay=decay,\
                                              epsilon=epsilon)
    return output
                                              
