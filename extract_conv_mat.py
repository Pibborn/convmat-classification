import tensorflow as tf
import argparse
import logging
import numpy as np
from tensorflow.python.platform import gfile
from models.research.slim.nets.vgg import vgg_d
from models.research.slim.nets import resnet_v1
from models.research.slim.nets import inception
from tensorflow.contrib.framework import arg_scope
import matplotlib.pyplot as plt

def extract_convmat_vgg():
    img_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])
    output, endpoints = vgg_d(img_ph)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint_path)
        names = [v.name for v in tf.all_variables() if 'conv' in v.name]
        names = set(['/'.join(name.split('/')[:-1]) for name in names])
        for name in names:
            with tf.variable_scope(name) as scope:
                tf.get_variable_scope().reuse_variables()
                weights = tf.get_variable('weights')
                weights = weights.eval()
                weights = np.reshape(weights, (3, 3, -1))
                mean_weights = np.mean(weights, axis=(0, 1))
                hist_weights = np.ones_like(mean_weights)/float(len(mean_weights))
                plt.hist(np.abs(mean_weights), weights=hist_weights, bins=30)
                plt.ylim((0, 0.6))
                filename = name.replace('/', '-')
                plt.savefig('hist/' + filename + '-weights.pdf')
                plt.clf()


def extract_convmat_inception():
    with arg_scope(inception.inception_v3_arg_scope()):
        img_ph = tf.placeholder(tf.float32, [None, 299, 299, 3])
        output, endpoints = inception.inception_v3(img_ph,
                                         num_classes=1001,
                                         is_training=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint_path)
        names = [v.name for v in tf.all_variables()]
        names = [v.name for v in tf.all_variables() if 'Conv' in v.name]
        names = set(['/'.join(name.split('/')[:-1]) for name in names])
        for name in names:
            with tf.variable_scope(name) as scope:
                tf.get_variable_scope().reuse_variables()
                try:
                    weights = tf.get_variable('weights')
                except ValueError:
                    continue
                weights = weights.eval()
                #if weights.shape[0:2] != (1, 1):
                flat_weights = np.reshape(weights, [weights.shape[0], weights.shape[1], -1])
                mean_weights = np.mean(flat_weights, axis=(0, 1))
                hist_weights = np.ones_like(mean_weights)/float(len(mean_weights))
                plt.hist(np.abs(mean_weights), weights=hist_weights, bins=30)
                plt.ylim((0, 0.8))
                filename = name.replace('/', '-')
                plt.savefig('hist/inception-v3/' + filename + '-' + str(weights.shape[0]) + 'x' + str(weights.shape[1]) + '-weights.pdf')
                plt.clf()


def extract_convmat_resnet():
    img_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])

    with arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(img_ph, 1000, is_training=False)
        saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint_path)
        names = [v.name for v in tf.all_variables()]
        names = [v.name for v in tf.all_variables() if 'conv' in v.name]
        names = set(['/'.join(name.split('/')[:-1]) for name in names])
        for name in names:
            with tf.variable_scope(name) as scope:
                tf.get_variable_scope().reuse_variables()
                try:
                    weights = tf.get_variable('weights')
                except ValueError:
                    continue
                weights = weights.eval()
                #if weights.shape[0:2] != (1, 1):
                flat_weights = np.reshape(weights, [weights.shape[0], weights.shape[1], -1])
                mean_weights = np.mean(flat_weights, axis=(0, 1))
                hist_weights = np.ones_like(mean_weights)/float(len(mean_weights))
                plt.hist(np.abs(mean_weights), weights=hist_weights, bins=30)
                plt.ylim((0, 0.8))
                filename = name.replace('/', '-')
                plt.savefig('hist/resnet-v1-50/' + filename + '-' + str(weights.shape[0]) + 'x' + str(weights.shape[1]) + '-weights.pdf')
                plt.clf()
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help = 'Path to the checkpoint file')
    args = parser.parse_args()
    
    checkpoint_filename = args.checkpoint_path.split('/')[-1]
    checkpoint_dirname = args.checkpoint_path.split('/')[:-1]
    
    if 'vgg' in checkpoint_filename:
        extract_convmat_vgg()

    if 'inception' in checkpoint_filename:
        extract_convmat_inception()

    if 'resnet' in checkpoint_filename:
        extract_convmat_resnet()
 
    
