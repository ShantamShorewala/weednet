import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os, sys
import numpy as np
import math
import skimage
import skimage.io
import skimage.transform
import cv2

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
IMAGE_DEPTH = 3

NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 180
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 30
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 3-D Tensor of [height, width, 1] type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=1,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads)
        #capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # tf.image_summary('images', images)

  return images, label_batch

def CamVid_reader_seq(filename_queue, seq_length):
  image_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[0])
  label_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[1])

  image_seq = []
  label_seq = []
  for im ,la in zip(image_seq_filenames, label_seq_filenames):
    imageValue = tf.read_file(tf.squeeze(im))
    labelValue = tf.read_file(tf.squeeze(la))
    image_bytes = tf.image.decode_image(imageValue, channels=3)
    label_bytes = tf.image.decode_png(labelValue)
    image = tf.cast(tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)), tf.float32)
    label = tf.cast(tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1)), tf.int64)
    image_seq.append(image)
    label_seq.append(label)
  return image_seq, label_seq

def CamVid_reader(filename_queue):

  image_filename = filename_queue[0]
  label_filename = filename_queue[1]

  imageValue = tf.read_file(image_filename)
  labelValue = tf.read_file(label_filename)

  #imageValue = tf.image.resize_images(imageValue, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
  #labelValue = tf.image.resize_images(labelValue, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])

  image_bytes = tf.image.decode_png(imageValue, channels=3)
  label_bytes = tf.image.decode_png(labelValue)

  image_bytes = tf.image.resize_images(image_bytes, [IMAGE_HEIGHT, IMAGE_WIDTH])
  label_bytes = tf.image.resize_images(label_bytes, [IMAGE_HEIGHT, IMAGE_WIDTH])
  print "Check Here"+str(image_bytes.shape)
  print "Check Here"+str(label_bytes.shape)

  image = tf.reshape(image_bytes, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
  label = tf.reshape(label_bytes, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])

  print "Image Size"+ str(image.shape)
  print "Label Size"+ str(label.shape)
  return image, label

def get_filename_list(path):
  fd = open(path)
  image_filenames = []
  label_filenames = []
  filenames = []
  for i in fd:
    i = i.strip().split(" ")
    image_filenames.append(i[0])
    label_filenames.append(i[1])
  return image_filenames, label_filenames

def CamVidInputs(image_filenames, label_filenames, batch_size):

  images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
  labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)
  filename_queue = tf.train.slice_input_producer([images, labels], shuffle=False)

  image, label = CamVid_reader(filename_queue)
  print "image return" +str(image.shape)
  # image = tf.reshape(image, [360,480,1][3])
  # label = tf.reshape(label, [360,480,1][3])
  reshaped_image = tf.cast(image, tf.float32)
  
  min_fraction_of_examples_in_queue = 0.5
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  min_queue_examples = 1
  print ('Filling queue with %d CamVid images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
  print "BATCH "+str(batch_size)
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
def get_all_test_data(im_list, la_list):
  images = []
  labels = []
  index = 0
  for im_filename, la_filename in zip(im_list, la_list):
    #im = np.array(skimage.io.imread(im_filename), np.float32)
    #la = np.array(skimage.io.imread(la_filename), np.float64)
    im = np.array(cv2.imread(im_filename), np.float32)
    la = np.array(cv2.imread(la_filename, 0), np.float64)
    print "1 " + str(im.shape)
    im = skimage.transform.resize(im, (240,320), preserve_range=True, mode='constant')
    la = skimage.transform.resize(la, (240,320), preserve_range=True, mode='constant')
    print "1 " + str(im.shape)
    im = im[np.newaxis,:]
    la = la[np.newaxis,:]
    #im = im[...,np.newaxis]
    la = la[...,np.newaxis]
    images.append(im)
    labels.append(la)
    #print im.shape
    #print la.shape
  return images, labels
