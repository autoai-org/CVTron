# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import re
import time
from datetime import datetime


class YoloTrainer(object):
    def __init__(self, dataset, net, moment, learning_rate, batch_size, image_size, max_objects, pretrain_path, train_dir, max_iter):
        self.net = net
        self.dataset = dataset
        self.moment = moment
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.width = image_size
        self.height = image_size
        self.max_objects = max_objects
        self.pretrain_path = pretrain_path
        self.train_dir = train_dir
        self.max_iter = max_iter

    def construct_graph(self):

        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(
            tf.float32, (self.batch_size, self.height, self.width, 3))
        self.labels = tf.placeholder(
            tf.float32, (self.batch_size, self.max_objects, 5))
        self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

        self.predicts = self.net.inference(self.images)
        self.total_loss, self.nilboy = self.net.loss(
            self.predicts, self.labels, self.objects_num)

        tf.summary.scalar('loss', self.total_loss)
        self.train_op = self.train()

    def train(self):
        pretrain_saver = tf.train.Saver(
            self.net.pretrained_collection, write_version=1)
        train_saver = tf.train.Saver(
            self.net.trainable_collection, write_version=1)
        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        sess.run(init)
        pretrain_saver.restore(sess, self.pretrain_path)
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

        for step in range(self.max_iter):
            start_time = time.time()
            np_images, np_labels, np_objects_num = self.dataset.batch()
            _, loss_value, result = sess.run([self.train_op, self.total_loss, self.nilboy], feed_dict={
                                             self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})
            duration = time.time() - start_time
            assert not np.isnan(loss_value)
            num_exmaples_per_step = self.dataset.batch_size
            examples_per_sec = num_exmaples_per_step / duration
            sec_per_batch = float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})
                summary_writer.add_summary(summary_str, step)
            if step % 5000 == 0:
                train_saver.save(sess, self.train_dir + '/model.ckpt', global_step=step)
        sess.close()