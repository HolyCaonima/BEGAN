
import math
import customDataGeter
import model
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
import prettytensor as pt

from scipy.misc import imsave

class GAN(object):

    def __init__(self, version, clip_abs, hidden_size, batch_size, learning_rate, data_directory, log_directory):
        '''GAN Construction function

        Args:
            hidden_size: the hidden size for random Value
            batch_size: the img num per batch
            learning_rate: the learning rate

        Returns:
            A tensor that expresses the encoder network

        Notify: output size dependence
        '''
        self.img_size = [64, 64]
        self.version = version
        self.clip_abs = clip_abs
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_directory = data_directory
        self.log_directory = log_directory

        # build the graph
        self._build_graph()
        self.merged_all = tf.summary.merge_all()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.log_directory, self.sess.graph)


    def _build_graph(self):
        self.k_ph = tf.placeholder(tf.float32, shape=[])
        # build up the hidden Z
        z = tf.truncated_normal([self.batch_size, self.hidden_size])
        input_img = customDataGeter.input(self.data_directory, self.img_size, self.batch_size)

        # the training step
        global_step = tf.Variable(0, trainable=False)

        # build template
        discriminator_template = model.build_discriminator_template(self.version, self.hidden_size)
        generator_template = model.build_generator_template(self.version, self.hidden_size)

        # instance the template
        self.g_out = generator_template.construct(input=z)
        real_disc_inst = discriminator_template.construct(input=input_img)
        fake_disc_inst = discriminator_template.construct(input=self.g_out)

        mu_real = tf.reduce_mean(tf.abs(real_disc_inst - input_img))
        mu_gen = tf.reduce_mean(tf.abs(fake_disc_inst - self.g_out))

        self.D_loss = mu_real - self.k_ph * mu_gen
        self.G_loss = mu_gen

        lam = 0.001
        gamma = 0.5
        self.out_k = self.k_ph + lam * (gamma * mu_real - mu_gen)
        self.convergence_measure = mu_real + np.abs(gamma * mu_real - mu_gen)

        # build the optimization operator (RMS no better than adam.)
        self.opt_d = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.D_loss, global_step,
                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator'))
        self.opt_g = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.G_loss, global_step,
                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator'))


    def update_params(self, current_step, in_k, d_step = 1, g_step = 1):
        # train citers 
        self.sess.run(self.opt_d, feed_dict={self.k_ph:in_k})
        self.sess.run(self.opt_g)
        ret_k = self.sess.run(self.out_k, feed_dict={self.k_ph:in_k})
        if ret_k>1:
            ret_k = 1.
        if ret_k<0:
            ret_k = 0.
        return ret_k

    def get_loss(self):
        measure = self.sess.run([self.convergence_measure])
        return measure

    def generate_and_save_images(self, num_samples, directory):
        '''Generates the images using the model and saves them in the directory

        Args:
            num_samples: number of samples to generate
            directory: a directory to save the images

        Notify: output size dependence
        '''
        imsize = self.img_size
        im_w = int(math.ceil(math.sqrt(num_samples)))
        big_img = np.zeros([im_w*imsize[1],im_w*imsize[0],3])
        imgs = self.sess.run(self.g_out)
        while imgs.shape[0]<num_samples:
            tmp = self.sess.run(self.g_out)
            imgs = np.concatenate((imgs, tmp), axis=0)
        for k in range(num_samples):
            slice_img = imgs[k].reshape(imsize[1], imsize[0], 3)
            big_img[(k/im_w)*imsize[1]:((k/im_w)+1)*imsize[1], (k%im_w)*imsize[0]:((k%im_w)+1)*imsize[0],:] = slice_img
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder) 
            imsave(os.path.join(imgs_folder, '%d.png') % k, slice_img)
        imsave(os.path.join(imgs_folder,"Agg.png"), big_img)
    
    def get_merged_image(self, num_samples):
        imsize = self.img_size
        im_w = int(math.ceil(math.sqrt(num_samples)))
        big_img = np.zeros([im_w*imsize[1],im_w*imsize[0],3])
        imgs = self.sess.run(self.g_out)
        while imgs.shape[0]<num_samples:
            tmp = self.sess.run(self.g_out)
            imgs = np.concatenate((imgs, tmp), axis=0)
        for k in range(num_samples):
            big_img[(k/im_w)*imsize[1]:((k/im_w)+1)*imsize[1], (k%im_w)*imsize[0]:((k%im_w)+1)*imsize[0],:] = imgs[k].reshape(imsize[1], imsize[0], 3)
        big_img = big_img.reshape([1,big_img.shape[0],big_img.shape[1],3])
        return big_img