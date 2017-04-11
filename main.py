

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf

from tensorflow.contrib import layers, losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave, imshow, imresize

from tensorflow.examples.tutorials.mnist import input_data

from progressbar import ProgressBar

from gan import GAN

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("global_step", 0, "the step of current training")
flags.DEFINE_integer("g_iter", 1, "iteration times of G per iteration")
flags.DEFINE_integer("d_iter", 1, "iteration times of D per iteration")
flags.DEFINE_float("clip_abs", 0.01, "clip of D")
flags.DEFINE_integer("batch_size", 32, "bathch size")
flags.DEFINE_integer("updates_per_epoch", 100, "update certain times then show the loss")
flags.DEFINE_integer("max_epoch", 500, "max epoch")
flags.DEFINE_string("working_directory", "./work", "the working directory of current job")
flags.DEFINE_string("data_directory", "./tmp/data", "directory of training data")
flags.DEFINE_integer("hidden_size", 128, "the size of hidden space")
flags.DEFINE_float("learning_rate", 0.00005, "learning rate")
flags.DEFINE_integer("version", 1, "the version of model")

FLAGS = flags.FLAGS

start_epoch = 0
k_p = 0.

def save_model(sess, saver):
    if not os.path.exists(os.path.join(FLAGS.working_directory,"save")):
        os.mkdir(os.path.join(FLAGS.working_directory,"save"))  
    if os.path.exists(os.path.join(FLAGS.working_directory,"save","desc")):
        os.remove(os.path.join(FLAGS.working_directory,"save","desc"))
    model_desc = open(os.path.join(FLAGS.working_directory,"save","desc"),'w')
    model_desc.write(str(FLAGS.global_step)+"\n")
    model_desc.write(str(FLAGS.g_iter)+"\n")
    model_desc.write(str(FLAGS.d_iter)+"\n")
    model_desc.write(str(FLAGS.clip_abs)+"\n")
    model_desc.write(str(FLAGS.batch_size)+"\n")
    model_desc.write(str(FLAGS.updates_per_epoch)+"\n")
    model_desc.write(str(FLAGS.max_epoch)+"\n")
    model_desc.write(str(FLAGS.data_directory)+"\n")
    model_desc.write(str(FLAGS.hidden_size)+"\n")
    model_desc.write(str(FLAGS.learning_rate)+"\n")
    model_desc.write(str(start_epoch)+"\n")
    model_desc.write(str(k_p)+"\n")
    saver.save(sess, os.path.join(FLAGS.working_directory,"save","model.data"))
    model_desc.close()
    print "model saved!"

def load_desc():
    if not os.path.exists(os.path.join(FLAGS.working_directory,"save", "desc")):
        print "model not exists!"
        return
    model_desc = open(os.path.join(FLAGS.working_directory,"save","desc"))
    FLAGS.global_step = int(model_desc.readline())
    FLAGS.g_iter = int(model_desc.readline())
    FLAGS.d_iter = int(model_desc.readline())
    FLAGS.clip_abs = float(model_desc.readline())
    FLAGS.batch_size = int(model_desc.readline())
    FLAGS.updates_per_epoch = int(model_desc.readline())
    FLAGS.max_epoch = int(model_desc.readline())
    FLAGS.data_directory = str(model_desc.readline()).replace("\n","")
    FLAGS.hidden_size = int(model_desc.readline())
    FLAGS.learning_rate = float(model_desc.readline())
    start_epoch = int(model_desc.readline())
    k_p = float(model_desc.readline())
    model_desc.close()

def load_model(sess, saver):
    if not os.path.exists(os.path.join(FLAGS.working_directory,"save", "desc")):
        print "model not exists!"
        return
    saver.restore(sess, os.path.join(FLAGS.working_directory,"save","model.data"))

def main():
    global k_p
    if not os.path.exists(FLAGS.working_directory):
        os.makedirs(FLAGS.working_directory)
    if not os.path.exists(FLAGS.data_directory):
        os.makedirs(FLAGS.data_directory)

    if os.path.exists(os.path.join(FLAGS.working_directory,"save","ver")):
        ver_desc = open(os.path.join(FLAGS.working_directory,"save","ver"))
        FLAGS.version = int(ver_desc.readline())
        ver_desc.close()
    else:
        if not os.path.exists(os.path.join(FLAGS.working_directory,"save")):
            os.mkdir(os.path.join(FLAGS.working_directory,"save"))
        ver_desc = open(os.path.join(FLAGS.working_directory,"save","ver"),'w')
        ver_desc.write(str(FLAGS.version)+"\n")
        ver_desc.close()

    load_desc()
    gan = GAN(FLAGS.version, FLAGS.clip_abs, FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.data_directory, os.path.join(FLAGS.working_directory, "log"))
    saver = tf.train.Saver()
    load_model(gan.sess, saver)

    # start the queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=gan.sess, coord=coord)
    
    for epoch in range(start_epoch, FLAGS.max_epoch):
        pbar = ProgressBar()
        for update in pbar(range(FLAGS.updates_per_epoch)):
            k_p = gan.update_params(FLAGS.global_step, k_p, FLAGS.d_iter, FLAGS.g_iter)
            FLAGS.global_step = FLAGS.global_step + 1
        
        cm = gan.get_loss()
        print "loss: " + str(cm)
        gan.generate_and_save_images(64, FLAGS.working_directory) #(int(math.sqrt(FLAGS.batch_size))+1)**2
        save_model(gan.sess, saver)

    # ask threads to stop
    coord.request_stop()

    # wait for threads to finish
    coord.join(threads)
    gan.sess.close()

if __name__ == '__main__':
    main()