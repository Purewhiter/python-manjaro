'''
Author      : PureWhite
Date        : 2021-01-26 23:16:01
LastEditors : PureWhite
LastEditTime: 2021-01-27 18:09:34
Description : 
'''

import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")