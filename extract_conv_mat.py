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
from numpy.lib.shape_base import apply_along_axis
import unittest

##Constants
RELATIVETOLL = 1e-2
ABSOLUTETOLL = 0

def plotting(names, shape='All'):
    print('Plotting...')
    for name in names:
        with tf.variable_scope(name) as scope:
            tf.get_variable_scope().reuse_variables()
            try:
                weights = tf.get_variable('weights')
            except ValueError:
                continue
            weights = weights.eval()
            if shape=='All' or weights.shape[0:2] == shape:
                print(name)
                flat_weights = np.reshape(weights, [weights.shape[0], weights.shape[1], -1])
            
                if compute_mean:
                    plot_weight_mean(flat_weights, name)
            
                if compute_sym and flat_weights.shape[0:2] == (3, 3):
                    plot_weight_symmetry(flat_weights, name)
            else:
                print('Skip...')
    print('Done.')

def plot_weight_mean(flat_weights, name):
    f_w_h = flat_weights.shape[0]
    f_w_w = flat_weights.shape[1] 
    mean_weights = np.mean(flat_weights, axis=(0, 1))
    hist_weights = np.ones_like(mean_weights)/float(len(mean_weights))
    _,ax = plt.subplots()
    ax.set_ylabel('Percentage')
    ax.set_title('Convolutional weights matrices - Absolute Mean')
    plt.hist(np.abs(mean_weights), weights=hist_weights, bins=30)
    plt.ylim((0, 0.8))
    filename = name.replace('/', '-')
    plt.savefig('hist/'+ plots_dirname + '/' + filename + '-' + str(f_w_h) + 'x' + str(f_w_w) + '-weights.pdf')
    plt.clf()
    
def plot_weight_symmetry(flat_weights, name):
    f_w_size = flat_weights.shape[0] #square size
    flat_weights = np.reshape(flat_weights,(f_w_size*f_w_size,-1))
    flat_weights = np.reshape(flat_weights.T,(-1,f_w_size,f_w_size))
    lenw = flat_weights.shape[0]
    s_ = np.zeros((lenw,4))
    s_[:,:] = [[is_sym(mat, type='HORIZONTAL'),is_sym(mat, type='VERTICAL'),is_sym(mat, type='DIAGSX'),is_sym(mat, type='DIAGDX')] for mat in flat_weights]
    s_= s_.T
    as_ = np.zeros((lenw,4))
    as_[:,:] = [[is_sym(mat, type='HORIZONTAL',asym=True),is_sym(mat, type='VERTICAL',asym=True),is_sym(mat, type='DIAGSX',asym=True),is_sym(mat, type='DIAGDX',asym=True)] for mat in flat_weights]
    as_=as_.T
    
    s_summary = np.sum(s_,axis=0)
    #more than 2 symmetry axes
    s_strong_count = 100*np.sum(s_summary >= 2)/lenw 
    
    #less than 2 symmetry axes
    s_weak_count = 100*np.sum(s_summary == 1)/lenw
    
    as_summary = np.sum(as_,axis=0)
    
    #Dovrebbero essere insiemi disgiunti, ma per tolleranze troppo alte considero simmetrie pari
    as_count = min(100*np.sum(as_ > 0)/lenw, 100 - s_strong_count - s_weak_count)
    
    #no symm
    none_count = 100 - s_strong_count - s_weak_count - as_count  
    
    
    _,ax = plt.subplots()
    ax.set_ylabel('Percentage')
    ax.set_title('Convolutional weights matrices - Symmetry')
    ax.set_xticklabels(('Multiple\nEven Symmetry', 'Single\nEven Symmetry', 'Odd\nSymmetry', 'None'), fontsize = 5.0)
    ax.set_xticks(list(range(4)))
    plt.bar(list(range(4)),[s_strong_count,s_weak_count,as_count,none_count])
    plt.ylim((0, 100))
    filename = name.replace('/', '-')
    plt.savefig('sym_0p01/'+ plots_dirname + '/' + filename + '-' + str(f_w_size) + 'x' + str(f_w_size) + '-sym.pdf')
    plt.clf()
    
    
    
def is_sym(mat, type='HORIZONTAL', asym=False):
    assert mat.shape == (3,3)
    rev_mat = np.copy(mat)
    if asym:
        rev_mat *= -1
    if type=='HORIZONTAL':
        rev_mat = np.flip(rev_mat, 0)
    if type=='VERTICAL':
        rev_mat = np.flip(rev_mat, 1)
    if type=='DIAGSX':
        rev_mat = rev_mat.T
    if type=='DIAGDX':
        rev_mat = rev_mat.T
        rev_mat = np.flip(rev_mat, 0)
        rev_mat = np.flip(rev_mat, 1)
    return np.allclose(mat, rev_mat, RELATIVETOLL, ABSOLUTETOLL)


def extract_convmat_vgg():
    img_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])
    output, endpoints = vgg_d(img_ph)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint_path)
        names = [v.name for v in tf.global_variables() if 'conv' in v.name]
        names = set(['/'.join(name.split('/')[:-1]) for name in names])
        plotting(names, shape=m_size)


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
        names = [v.name for v in tf.global_variables() if 'Conv' in v.name]
        names = set(['/'.join(name.split('/')[:-1]) for name in names])
        plotting(names, shape=m_size)
                
                    


def extract_convmat_resnet():
    img_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])

    with arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(img_ph, 1000, is_training=False)
        saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint_path)
        names = [v.name for v in tf.global_variables() if 'conv' in v.name]
        names = set(['/'.join(name.split('/')[:-1]) for name in names])
        plotting(names, shape=m_size)

class TestSymCheckFunctions(unittest.TestCase):

    def test_even(self):
        a = np.array([[1, 2, 1], [5, 5, 5], [1,2,1]])
        self.assertTrue(is_sym(a, type='HORIZONTAL'))
        self.assertTrue(is_sym(a, type='VERTICAL'))
        self.assertTrue(is_sym(a, type='DIAGSX'))
        self.assertTrue(is_sym(a, type='DIAGDX'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help = 'Path to the checkpoint file')
    parser.add_argument('-dm','--disable_mean', help = 'Disable mean plots', action="store_true")
    parser.add_argument('-ds','--disable_sym', help = 'Disable symmetry plots', action="store_true")
    parser.add_argument('-m3x3','--m3x3', help = 'Plot only for 3x3 weights matrices', action="store_true")
    args = parser.parse_args()
    
    checkpoint_filename = args.checkpoint_path.split('/')[-1]
    checkpoint_dirname = args.checkpoint_path.split('/')[:-1]
    plots_dirname = '.'.join(checkpoint_filename.split('.')[:-1])
    
    compute_mean = True 
    compute_sym = True
    m_size = 'All'
    if args.disable_mean:
        compute_mean = False
    
    if args.disable_sym:
        compute_sym = False
        
    if args.m3x3:
        m_size = (3,3)
    
    if 'vgg' in checkpoint_filename:
        extract_convmat_vgg()

    if 'inception' in checkpoint_filename:
        extract_convmat_inception()

    if 'resnet' in checkpoint_filename:
        extract_convmat_resnet()
 
    
