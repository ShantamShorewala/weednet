import os, sys
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import math
import skimage
import skimage.io
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
from Inputs import *

image_h=240
image_w=320
image_c=3

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1 # Learning rate decay factor.

num_train_examples=50
initial_lr=0.001
batch_size=1
num_classes=3
max_steps=10000
log_dir = '/path to file/logs'#location for storing log files
image_dir = '/path to file/trainNdvi.txt' #path to dataset 

#----------------------------------------------------------------------------------------------

def msra_initializer(kl, dl):
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def orthogonal_initializer(scale = 1.1):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
    	flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
	return _initializer


def loss(logits, labels):
	global num_classes
	logits = tf.reshape(logits, (-1,num_classes))
	labels = tf.reshape(labels, [-1])
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon
        print logits.shape
        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))
        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
        print labels.shape
        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

def cal_loss(logits, labels):
    loss_weight = np.array([
      0.7,
      1.2,
      1.2]) # class 0~2

    labels = tf.cast(labels, tf.int32)
    return weighted_loss(logits, labels, num_classes=3, head=loss_weight)

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
    	kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
        	conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
        	conv_out = batch_norm_layer(bias, train_phase, scope.name)
	return conv_out

#------------------------------------------------------------------------------------------------------------

def get_deconv_filter(f_shape):
    width = f_shape[0]
    height = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
        	value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        	bilinear[x, y] = value
  	weights = np.zeros(f_shape)
  	for i in range(f_shape[2]):
  		weights[:, :, i, i] = bilinear

  	init = tf.constant_initializer(value=weights, dtype=tf.float32)
  	return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  # sess_temp = tf.InteractiveSession()
  	sess_temp = tf.global_variables_initializer()
  	strides = [1, stride, stride, 1]
  	with tf.variable_scope(name):
  		weights = get_deconv_filter(f_shape)
  		deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,strides=strides, padding='SAME')
	return deconv

def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,updates_collections=None, center=False, scope=scope+"_bn", reuse = True))

def orthogonal_initializer(scale = 1.1):
    
    def initializer(shape, dtype=tf.float32, partition_info=None):
	    flat_shape = (shape[0], np.prod(shape[1:]))
	    a = np.random.normal(0.0, 1.0, flat_shape)
	    u, _, v = np.linalg.svd(a, full_matrices=False)
	    # pick the one with the correct shape
	    q = u if u.shape == flat_shape else v
	    q = q.reshape(shape) #this needs to be corrected to float32
	    return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return initializer

def residual_block(inputT, name, phase_train, uid):
	conv_one = conv_layer_with_bn(inputT, [1, 1, 64, 32], phase_train, name="conv_one"+uid)
	conv_two = conv_layer_with_bn(conv_one, [5, 1, 32, 32], phase_train, name="conv_two"+uid)
	conv_three = conv_layer_with_bn(conv_two, [1, 5, 32, 32], phase_train, name="conv_three"+uid)
	conv_four = conv_layer_with_bn(conv_three, [1, 1, 32, 64], phase_train, name="conv_four"+uid)
	conv_final = tf.add(inputT, conv_four)
	return conv_final


def residual_block_2(inputT, name, phase_train, uid):	 
	conv_one = conv_layer_with_bn(inputT, [1, 1, 64, 32], phase_train, False,  name="conv_one"+uid)
	conv_two = conv_layer_with_bn(conv_one, [5, 1, 32, 32], phase_train, False, name="conv_two"+uid)
	conv_three = conv_layer_with_bn(conv_two, [1, 5, 32, 32], phase_train, False, name="conv_three"+uid)
	conv_four = conv_layer_with_bn(conv_three, [1, 1, 32, 64], phase_train, False, name="conv_four"+uid)
	conv_final = tf.add(inputT, conv_four)
	return conv_final


def inference(images, labels, phase_train):
	global num_classes, batch_size

	#local normalization on input images
	norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')

	#convolve input layer
	conv1 = conv_layer_with_bn(norm1, [5, 5, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
	print conv1.shape
	#residual set 1
	conv2 = residual_block(conv1, "conv2", phase_train, "1")
	conv3 = residual_block(conv2, "conv3", phase_train, "2")
	conv4 = residual_block(conv3, "conv4", phase_train, "3")

	pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
	print pool1.shape
	#residual block 2
	conv5 = residual_block(pool1, "conv5", phase_train, "4")
	conv6 = residual_block(conv5, "conv6", phase_train, "5")
	conv7 = residual_block(conv6, "conv7", phase_train, "6")

	pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
	print pool2.shape
	#residual block 3

	conv8 = residual_block(pool2, "conv8", phase_train, "7")
	conv9 = residual_block(conv8, "conv9", phase_train, "8")
	conv10 = residual_block(conv9, "conv10", phase_train, "9")

	pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
	print pool3.shape
	#residual block 4

	conv11 = residual_block(pool3, "conv11", phase_train, "10")
	conv12 = residual_block(conv11, "conv12", phase_train, "11")
	conv13 = residual_block(conv12, "conv13", phase_train, "12")

	pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
	print pool4.shape
	#end of encoder - start of decoder
	
	#residual block 5
	upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 30, 40, 64], 2, "up4")
	conv_decode14 = residual_block_2(upsample4, "conv_decode14", phase_train, "13")
	conv_decode15 = residual_block_2(conv_decode14, "conv_decode15", phase_train, "14")
	conv_decode16 = residual_block_2(conv_decode15, "conv_decode16", phase_train, "15")

	#residual block 6
	upsample3 = deconv_layer(conv_decode16, [2, 2, 64, 64], [batch_size, 60, 80, 64], 2, "up3")
	conv_decode17 = residual_block_2(upsample3, "conv_decode17", phase_train, "16")
	conv_decode18 = residual_block_2(conv_decode17, "conv_decode18", phase_train, "17")
	conv_decode19 = residual_block_2(conv_decode18, "conv_decode19", phase_train, "18")
	
	#residual block 7
	upsample2 = deconv_layer(conv_decode19, [2, 2, 64, 64], [batch_size, 120, 160, 64], 2, "up2")
	conv_decode20 = residual_block_2(upsample2, "conv_decode20", phase_train, "19")
	conv_decode21 = residual_block_2(conv_decode20, "conv_decode20", phase_train, "20")
	conv_decode22 = residual_block_2(conv_decode21, "conv_decode21", phase_train, "21")

	#residual block 8
	upsample1 = deconv_layer(conv_decode22, [2, 2, 64, 64], [batch_size, 240, 320, 64], 2, "up1")
	conv_decode23 = residual_block_2(upsample1, "conv_decode23", phase_train, "22")
	conv_decode24 = residual_block_2(conv_decode23, "conv_decode24", phase_train, "23")
	conv_decode25 = residual_block_2(conv_decode24, "conv_decode25", phase_train, "24")
	print conv_decode25.shape
	#conv_decode26 = conv_layer_with_bn(conv_decode25, [5, 5, 64, images.get_shape().as_list()[3]], phase_train, name="conv26")
	#print conv_decode26.shape
	#start classification
	with tf.variable_scope('conv_classifier') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[1, 1, 64, num_classes], initializer=msra_initializer(1, 64),wd=0.0005)
		print kernel.shape
		conv = tf.nn.conv2d(conv_decode25, kernel, [1, 1, 1, 1], padding='SAME')
		print "final" + str(conv.shape)
		biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))
		conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

	logit = conv_classifier
	loss = cal_loss(conv_classifier, labels)
	return loss, logit

def train(total_loss, global_step):
	global initial_lr
	lr = initial_lr
	loss_averages_op = _add_loss_summaries(total_loss)
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.AdamOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

   	for grad, var in grads:
   		if grad is not None:
   			tf.summary.histogram(var.op.name + '/gradients', grad)

   	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
   	variables_averages_op = variable_averages.apply(tf.trainable_variables())

   	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
   		train_op = tf.no_op(name='train')

	return train_op
  
#--------------------------------------------------------------------------------------------------------------

def model(one):
	global max_steps, batch_size, image_w, image_h, image_c, image_dir, log_dir, log_dir
	startstep=0
	image_filenames, label_filenames = get_filename_list(image_dir)

	with tf.Graph().as_default():
		train_data_node = tf.placeholder( tf.float32, shape=[batch_size, image_h, image_w, image_c])
		train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])
		phase_train = tf.placeholder(tf.bool, name='phase_train')
		global_step = tf.Variable(0, trainable=False)

		images, labels = CamVidInputs(image_filenames, label_filenames, batch_size)
		loss, eval_prediction = inference(train_data_node, train_labels_node, phase_train)
		train_op = train(loss, global_step)
		# print "image"+str(image.shape)
		# print "label"+str(label.shape)
		saver = tf.train.Saver(tf.global_variables())
		summary_op = tf.summary.merge_all()

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
		
		with tf.Session(config = config) as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
	       		average_pl = tf.placeholder(tf.float32)
	        	acc_pl = tf.placeholder(tf.float32)
	        	iu_pl = tf.placeholder(tf.float32)
	        	average_summary = tf.summary.scalar("test_average_loss", average_pl)
	        	acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
	        	iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

	        	for step in range(startstep, startstep + max_steps):
	        		image_batch ,label_batch = sess.run([images, labels])
	        # since we still use mini-batches in validation, still set bn-layer phase_train = True
	        		feed_dict = {train_data_node: image_batch, train_labels_node: label_batch, phase_train: True}
	
		        	start_time = time.time()
		        	_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
	        		duration = time.time() - start_time
	        		assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

	        		if step % 10 == 0:
	        			num_examples_per_step = batch_size
	        			examples_per_sec = num_examples_per_step / duration
	        			sec_per_batch = float(duration)

		        		format_str = ('%s: step %d, loss = %.2f(%.1f examples/sec; %.3f ' 'sec/batch)')
		        		print (format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))

	          # eval current training batch pre-class accuracy
		          		pred = sess.run(eval_prediction, feed_dict=feed_dict)
		          		per_class_acc(pred, label_batch)

					if step % 1000 == 0 or (step + 1) == max_steps:
						checkpoint_path = os.path.join(log_dir, 'model.ckpt')
						saver.save(sess, checkpoint_path, global_step=step)
			coord.request_stop()
			coord.join(threads)

print('The model is set to Training')
print("Max training Iteration: %d"%max_steps)
print("Initial lr: %f"%initial_lr)
print("CamVid Image dir: %s"%image_dir)
print("Batch Size: %d"%batch_size)
print("Log dir: %s"%log_dir)

tf.app.run(main = model)
