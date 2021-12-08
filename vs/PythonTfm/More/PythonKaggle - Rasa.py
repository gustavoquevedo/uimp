

import tensorflow_hub as hub

import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()


embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

embedding = embed(["Hello World!"])print('a')