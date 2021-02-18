'''
Author      : PureWhite
Date        : 2021-01-26 23:16:01
LastEditors : PureWhite
LastEditTime: 2021-01-27 15:54:32
Description : 
'''
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant(1.)
b = tf.constant(3.)
print(a+b)

print('GPU:', tf.test.is_gpu_available())