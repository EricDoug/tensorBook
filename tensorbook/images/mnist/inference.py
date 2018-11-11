# _*_ coding:utf-8 _*_
# @Time: 2018/11/11 下午7:19
# @Author: EricDoug
# @File: inference.py

import tensorflow as tf


INPUT_NODE = 784  # 输入层的节点数
OUTPUT_NODE = 10  # 输出层的节点数
LAYER1_NODE = 500  # 隐藏层1的节点数



def get_weight_variable(shape, regularizer):

    weights = tf.get_variable(
        "weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights

def inference(input_tensor, reqularizer):

    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], reqularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], reqularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


