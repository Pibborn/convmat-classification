import tensorflow as tf
import argparse
import logging
from tensorflow.python.platform import gfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pb_path', help = 'Path to the .pb model file')
    args = parser.parse_args()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pb_filename = args.pb_path.split('/')[-1]
        pb_dirname = args.pb_path.split('/')[:-1]
        
        with gfile.FastGFile(args.pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
        
        tf.summary.FileWriter('./graphs/' + pb_filename + '-graph', sess.graph)
        logging.info('Dumped graph at graphs/' + pb_filename)
        
