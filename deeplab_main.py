import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import time
import glob
import nibabel
import os
from scipy.misc import imsave, imread

import deeplab_model as deeplab

################## modified on 18/03/26 after segnet works

DATAPATH = "\\\\hnascifs01.uwhis.hosp.wisc.edu\\einas01\\Groups\\" +  \
       "PETMR\\deepMRAC_pelvis\\training_data20170927_resliced_augmented_corrected_nii"
A_NAME = "axt2-???.nii"
B_NAME = "mask-???.nii"
EPOCH = 210
BATCHSIZE = 10
DATALENGTH = 1000
LEARNING_RATE = 0.0002
LOOPS = DATALENGTH // BATCHSIZE
IMGtoPATH = "./my_output_test0326_2-{}/"
CheckPointPATH = "./ckpt_test0326/my_model.ckpt"



# check model
#fake_img = tf.constant(np.random.normal(size=[4,160,256,1]),dtype=tf.float32)
#fake_label = tf.constant(np.random.choice(4,size=[4,48,256,3]))
#fake_out, _ , ss= deeplab.deeplab_v3(fake_img,"deeplab")

def loss_calc(logits, labels, class_inc_bg):
    labels = labels[...,0]
    onehot_labels = tf.one_hot(labels, class_inc_bg)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    loss = tf.reduce_mean(unweighted_losses)
    return loss

##def loss_calc(result, labels, class_inc_bg):
##    labels = labels[:,:,:,0]
##    onehot_labels = tf.one_hot(labels, class_inc_bg)
##    unweighted_losses = -tf.log(result)*onehot_labels
##    loss = tf.reduce_mean(unweighted_losses)
##    return loss

#check loss
#fake_label = tf.constant(np.random.choice(4,size=[4,160,256,1]))
#check_loss = loss_calc(fake_out, fake_label, 4)

def train(data_path, num_of_class,from_epch=0):
    
    tf.set_random_seed(10)
    imgs = tf.placeholder(tf.float32,[None,160,256,1])
    #bchsz =  tf.shape(imgs)[0]
    #eval_img = tf.placeholder(tf.float32, [1,160,256,1])
    labels = tf.placeholder(tf.uint8, [None,160,256,1])
    lr = tf.placeholder(tf.float32, name="learning_rate")
    is_training = tf.placeholder(tf.bool, name="is_training")
    
    logits, pred = deeplab.deeplab_v3(imgs,"deeplabv3",is_training)
    eval_out = tf.argmax(logits, axis=-1)
    loss = loss_calc(logits, labels, num_of_class)
    optimizer = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss)
    saver = tf.train.Saver()
    train_imgs = (prepare_data(data_path,A_NAME)+1.82)/5.1-1
    train_label = prepare_data(data_path,B_NAME,label=True)
    test_imgs = train_imgs[DATALENGTH:,:,:,:]
    test_label = train_label[DATALENGTH:,:,:,:]
    with tf.Session() as sess:
        if from_epch==0:
            sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCH):
            lrate = LEARNING_RATE * (1 - epoch/EPOCH)**0.8
            for i in range(LOOPS):
                tic = time.time()
                img_bch = train_imgs[(i*BATCHSIZE):((i+1)*BATCHSIZE),:]
                label_bch = train_label[(i*BATCHSIZE):((i+1)*BATCHSIZE),:]
                _,cur_loss = sess.run([train_step,loss],feed_dict={imgs:img_bch,
                                is_training:True, labels:label_bch,lr:lrate})
                #cur_loss = sess.run(loss,feed_dict={imgs:img_bch, is_training:False,
                #                   labels:label_bch})
                print("Epoch {}: BATCH {} Time: {:.4f} Loss:{:.4f} ".format(
                    epoch,i,time.time()-tic, cur_loss))
            if epoch % 50 == 9 and i == LOOPS - 1:
                saver.save(sess, CheckPointPATH, global_step = epoch)
                if not os.path.exists(IMGtoPATH.format(epoch)):
                    os.mkdir(IMGtoPATH.format(epoch))
                for j in range(test_imgs.shape[0]//BATCHSIZE):
                    out = sess.run(eval_out, feed_dict={is_training: True,
                        imgs:test_imgs[j*BATCHSIZE:(j+1)*BATCHSIZE,:,:,:]})
                    out_f = sess.run(eval_out, feed_dict={is_training: False,
                        imgs:test_imgs[j*BATCHSIZE:(j+1)*BATCHSIZE,:,:,:]})
                    for i in range(BATCHSIZE):
                        concat_out = np.concatenate((out_f[i,:,:],
                            out[i,:,:], test_label[j*BATCHSIZE+i,:,:,0]),axis=0)
                        imsave(os.path.join(IMGtoPATH.format(epoch), \
                                        "out{}.jpg".format(j*BATCHSIZE+i)),concat_out)
                    

def prepare_data(path,name,label=False):
    ## there is some issue in order using glob.glob, so use sorted()
    imglist = []
    nii_names = sorted(glob.glob(os.path.join(path,name)))
    for nii_name in nii_names:
        img_slice = nibabel.load(nii_name).get_data()[6:166,:,:]
        for j in range(img_slice.shape[2]):
            if np.max(img_slice[:,:,j:(j+1)]) != np.min(img_slice[:,:,j:(j+1)]):
                imglist.append(np.array(img_slice[:,:,j:(j+1)]))
    if label==False:
        return np.array(imglist,dtype=np.float32)
    else:
        return np.array(imglist,dtype=np.uint8)

if __name__ == "__main__":
    train(DATAPATH,4,0)

