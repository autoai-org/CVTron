import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim
from SSD_settings import *

def SSDHook(feature_map, hook_id):
	"""
	Takes input feature map, output the predictions tensor
	hook_id is for variable_scope unqie string ID
	"""
	with tf.variable_scope('ssd_hook_' + hook_id):
		# Note we have linear activation (i.e. no activation function)
		net_conf = slim.conv2d(feature_map, NUM_PRED_CONF, [3, 3], activation_fn=None, scope='conv_conf')
		net_conf = tf.contrib.layers.flatten(net_conf)

		net_loc = slim.conv2d(feature_map, NUM_PRED_LOC, [3, 3], activation_fn=None, scope='conv_loc')
		net_loc = tf.contrib.layers.flatten(net_loc)

	return net_conf, net_loc

def ModelHelper(y_pred_conf, y_pred_loc):
	"""
	Define loss function, optimizer, predictions, and accuracy metric
	Loss includes confidence loss and localization loss
	conf_loss_mask is created at batch generation time, to mask the confidence losses
	It has 1 at locations w/ positives, and 1 at select negative locations
	such that negative-to-positive ratio of NEG_POS_RATIO is satisfied
	Arguments:
		* y_pred_conf: Class predictions from model,
			a tensor of shape [batch_size, num_feature_map_cells * num_defaul_boxes * num_classes]
		* y_pred_loc: Localization predictions from model,
			a tensor of shape [batch_size, num_feature_map_cells * num_defaul_boxes * 4]
	Returns relevant tensor references
	"""
	num_total_preds = 0
	for fm_size in FM_SIZES:
		num_total_preds += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
	num_total_preds_conf = num_total_preds * NUM_CLASSES
	num_total_preds_loc  = num_total_preds * 4

	# Input tensors
	y_true_conf = tf.placeholder(tf.int32, [None, num_total_preds], name='y_true_conf')  # classification ground-truth labels
	y_true_loc  = tf.placeholder(tf.float32, [None, num_total_preds_loc], name='y_true_loc')  # localization ground-truth labels
	conf_loss_mask = tf.placeholder(tf.float32, [None, num_total_preds], name='conf_loss_mask')  # 1 mask "bit" per def. box

	# Confidence loss
	logits = tf.reshape(y_pred_conf, [-1, num_total_preds, NUM_CLASSES])
	conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_true_conf)
	conf_loss = conf_loss_mask * conf_loss  # "zero-out" the loss for don't-care negatives
	conf_loss = tf.reduce_sum(conf_loss)

	# Localization loss (smooth L1 loss)
	# loc_loss_mask is analagous to conf_loss_mask, except 4 times the size
	diff = y_true_loc - y_pred_loc
	
	loc_loss_l2 = 0.5 * (diff**2.0)
	loc_loss_l1 = tf.abs(diff) - 0.5
	smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
	loc_loss = tf.select(smooth_l1_condition, loc_loss_l2, loc_loss_l1)
	
	loc_loss_mask = tf.minimum(y_true_conf, 1)  # have non-zero localization loss only where we have matching ground-truth box
	loc_loss_mask = tf.to_float(loc_loss_mask)
	loc_loss_mask = tf.stack([loc_loss_mask] * 4, axis=2)  # [0, 1, 1] -> [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], ...]
	loc_loss_mask = tf.reshape(loc_loss_mask, [-1, num_total_preds_loc])  # removing the inner-most dimension of above
	loc_loss = loc_loss_mask * loc_loss
	loc_loss = tf.reduce_sum(loc_loss)

	# Weighted average of confidence loss and localization loss
	# Also add regularization loss
	loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss + tf.reduce_sum(slim.losses.get_regularization_losses())
	optimizer = OPT.minimize(loss)

	#reported_loss = loss #tf.reduce_sum(loss, 1)  # DEBUG

	# Class probabilities and predictions
	probs_all = tf.nn.softmax(logits)
	probs, preds_conf = tf.nn.top_k(probs_all)  # take top-1 probability, and the index is the predicted class
	probs = tf.reshape(probs, [-1, num_total_preds])
	preds_conf = tf.reshape(preds_conf, [-1, num_total_preds])

	# Return a dictionary of {tensor_name: tensor_reference}
	ret_dict = {
		'y_true_conf': y_true_conf,
		'y_true_loc': y_true_loc,
		'conf_loss_mask': conf_loss_mask,
		'optimizer': optimizer,
		'conf_loss': conf_loss,
		'loc_loss': loc_loss,
		'loss': loss,
		'probs': probs,
		'preds_conf': preds_conf,
		'preds_loc': y_pred_loc,
	}
	return ret_dict