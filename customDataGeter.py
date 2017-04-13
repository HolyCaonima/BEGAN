
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imsave, imshow, imread

def read_and_decode2(filename_queue, img_size):   
    reader = tf.WholeFileReader()
    _, val = reader.read(filename_queue)
    # decode the png image
    image = tf.image.decode_jpeg(val, channels=3)

    # Convert to float image
    image = tf.cast(image, tf.float32)

    image.set_shape((64, 64, 3))
    #image = tf.reshape(image, [1, 64, 64, 3])
    #image = tf.image.resize_images(image,[32, 32])
    #image = tf.reshape(image, [32, 32, 3])

    # normalize
    image = image * (2. / 255.) - 1

    return image

def input(img_directory, img_size, batch_size):
    filenames = tf.train.match_filenames_once(os.path.join(img_directory, '*.jpg'))
    filename_queue = tf.train.string_input_producer(filenames)
    image = read_and_decode2(filename_queue, img_size)

    # randomly flip
    image = tf.image.random_flip_left_right(image)

    num_preprocess_threads = 4

    # ensure that the random shuffling has good mixing properties
    min_queue_examples = 20000
    format_str = ('Filling queue with {} images before training. '
                  'This might take a while.')
    print(format_str.format(min_queue_examples))

    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=3*min_queue_examples,
        min_after_dequeue=min_queue_examples)

    return images