import tensorflow as tf
import argparse
import logging
import numpy as np
from tensorflow.python.platform import gfile
from models.research.slim.nets.vgg import vgg_d
from models.research.slim.nets import resnet_v1
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
                print(weights.shape)
                #flat_weights = np.reshape(weights, [-1, 3, 3])
                #mean_weights = np.mean(flat_weights, axis=(1, 2))
                #hist_weights = np.ones_like(mean_weights)/float(len(mean_weights))
                #plt.hist(np.abs(mean_weights), weights=hist_weights, bins=20)
                #plt.ylim(0, 0.6)
                #filename = name.replace('/', '-')
                #plt.savefig('hist/' + filename + '-weights.pdf')
                #plt.clf()


def extract_convmat_inception():
    img_ph = tf.placeholder(tf.float32, [None, 299, 299, 3])
    with arg_scope(models.research.slim.inception.inception_v3_arg_scope()):
        output, endpoints = inception_v3(img_ph,
                                         num_classes=1001,
                                         is_training=False)
        print('ehi')

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
        print(names)
        for name in names:
            with tf.variable_scope(name) as scope:
                tf.get_variable_scope().reuse_variables()
                try:
                    weights = tf.get_variable('weights')
                except ValueError:
                    print(name)
                    continue
                weights = weights.eval()
                if weights.shape[0:2] == (3, 3):
                    flat_weights = np.reshape(weights, [-1, 3, 3])
                    mean_weights = np.mean(flat_weights, axis=(1, 2))
                    hist_weights = np.ones_like(mean_weights)/float(len(mean_weights))
                    plt.hist(np.abs(mean_weights), weights=hist_weights, bins=20)
                    plt.ylim(0, 0.6)
                    filename = name.replace('/', '-')
                    plt.savefig('hist/' + filename + '-weights.pdf')
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
 
    
