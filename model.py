import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
import prettytensor as pt
import ops

def build_discriminator_template(version, hidden_size):
    num_filters = 64
    with tf.variable_scope('discriminator'):
        discriminator = pt.template('input')
        discriminator = discriminator.conv2d(5, num_filters, tf.nn.elu)
        discriminator = discriminator.conv2d(5, num_filters, tf.nn.elu, [2, 2], batch_normalize=True)

        discriminator = discriminator.conv2d(5, num_filters*2, tf.nn.elu, batch_normalize=True)
        discriminator = discriminator.conv2d(5, num_filters*2, tf.nn.elu, [2, 2], batch_normalize=True)

        discriminator = discriminator.conv2d(5, num_filters*4, tf.nn.elu, batch_normalize=True)
        discriminator = discriminator.conv2d(5, num_filters*4, tf.nn.elu, [2, 2], batch_normalize=True)

        discriminator = discriminator.conv2d(5, num_filters*8, tf.nn.elu, batch_normalize=True)
        discriminator = discriminator.conv2d(5, num_filters*8, tf.nn.elu, [2, 2], batch_normalize=True)

        # flatten for fc
        discriminator = discriminator.flatten()
        discriminator = discriminator.fully_connected(hidden_size).apply(tf.nn.elu)

        discriminator = discriminator.fully_connected(4*4*512).apply(tf.nn.elu)
        discriminator = discriminator.reshape([-1, 4, 4, 512])
        
        discriminator = discriminator.conv2d(5, 512, tf.nn.elu, batch_normalize=True)
        discriminator = discriminator.upsample_conv(5, 512).batch_normalize().apply(tf.nn.elu)

        discriminator = discriminator.conv2d(5, 256, tf.nn.elu, batch_normalize=True)
        discriminator = discriminator.upsample_conv(5, 256).batch_normalize().apply(tf.nn.elu)

        discriminator = discriminator.conv2d(5, 128, tf.nn.elu, batch_normalize=True)
        discriminator = discriminator.upsample_conv(5, 128).batch_normalize().apply(tf.nn.elu)

        discriminator = discriminator.conv2d(5, 64, tf.nn.elu, batch_normalize=True)
        discriminator = discriminator.upsample_conv(5, 64).batch_normalize().apply(tf.nn.elu)

        discriminator = discriminator.conv2d(5, 3).apply(tf.sigmoid)
        
    return discriminator

def build_generator_template(version, hidden_size):
    with tf.variable_scope('generator'):
        generator = pt.template('input')

        generator = generator.fully_connected(4*4*512).apply(tf.nn.elu)

        generator = generator.reshape([-1, 4, 4, 512])

        generator = generator.conv2d(5, 512, tf.nn.elu, batch_normalize=True)
        generator = generator.upsample_conv(5, 512).batch_normalize().apply(tf.nn.elu)

        generator = generator.conv2d(5, 256, tf.nn.elu, batch_normalize=True)
        generator = generator.upsample_conv(5, 256).batch_normalize().apply(tf.nn.elu)

        generator = generator.conv2d(5, 128, tf.nn.elu, batch_normalize=True)
        generator = generator.upsample_conv(5, 128).batch_normalize().apply(tf.nn.elu)

        generator = generator.conv2d(5, 64, tf.nn.elu, batch_normalize=True)
        generator = generator.upsample_conv(5, 64).batch_normalize().apply(tf.nn.elu)

        generator = generator.conv2d(5, 3).apply(tf.sigmoid)

    return generator
