import os
import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(THIS_DIR, '..')

if __name__ == '__main__':
    pb_list = glob.glob(os.path.join(ROOT_DIR, 'inf_graph', '*.pb'))
    for pb_file in pb_list:
        subprocess.run(['bazel-bin/tensorflow/python/tools/freeze_graph',
      '--input_graph=' + os.path.join(ROOT_DIR, pb_file),
      '--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt',
      '--input_binary=true', '--output_graph=/tmp/frozen_inception_v3.pb',
      '--output_node_names=InceptionV3/Predictions/Reshape_1'])

