import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
import prettytensor as pt
import ops

def build_discriminator_template(version, hidden_size):
    num_filters = 64
    with tf.variable_scope('discriminator'):
        discriminator = pt.template('input')
        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu, [2, 2])

        discriminator = discriminator.conv2d(3, num_filters*2, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters*2, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters*2, tf.nn.elu, [2, 2])

        discriminator = discriminator.conv2d(3, num_filters*3, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters*3, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters*3, tf.nn.elu, [2, 2])

        discriminator = discriminator.conv2d(3, num_filters*4, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters*4, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters*4, tf.nn.elu, [2, 2])

        # flatten for fc
        discriminator = discriminator.flatten()
        discriminator = discriminator.fully_connected(hidden_size)

        discriminator = discriminator.fully_connected(8*8*num_filters)
        discriminator = discriminator.reshape([-1, 8, 8, num_filters])
        
        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.upsample2x()

        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.upsample2x()

        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.upsample2x()

        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)
        discriminator = discriminator.conv2d(3, num_filters, tf.nn.elu)

        discriminator = discriminator.conv2d(3, 3)
        
    return discriminator

def build_generator_template(version, hidden_size):
    num_filters = 64
    with tf.variable_scope('generator'):
        generator = pt.template('input')

        generator = generator.fully_connected(8*8*num_filters)

        generator = generator.reshape([-1, 8, 8, num_filters])

        generator = generator.conv2d(3, num_filters, tf.nn.elu)
        generator = generator.conv2d(3, num_filters, tf.nn.elu)
        generator = generator.upsample2x()

        generator = generator.conv2d(3, num_filters, tf.nn.elu)
        generator = generator.conv2d(3, num_filters, tf.nn.elu)
        generator = generator.upsample2x()

        generator = generator.conv2d(3, num_filters, tf.nn.elu)
        generator = generator.conv2d(3, num_filters, tf.nn.elu)
        generator = generator.upsample2x()

        generator = generator.conv2d(3, num_filters, tf.nn.elu)
        generator = generator.conv2d(3, num_filters, tf.nn.elu)

        generator = generator.conv2d(3, 3)

    return generator
