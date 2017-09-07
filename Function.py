import tensorflow as tf
import random
import math


def new_weights(shape):  
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))    # 初始化为随机值
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))          # 初始化为常数
def new_conv_layer(input,              # 前一层.
                  num_input_channels, # 前一层通道数
                  filter_size,        # 卷积核尺寸
                  num_filters,        # 卷积核数目
                  use_pooling=True,
                  use_Abs=True):  # 使用 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    # 根据跟定形状创建权重
    weights = new_weights(shape=shape)
    # 创建新的偏置，每个卷积核一个偏置
    biases = new_biases(length=num_filters)

    # 创建卷积层。注意stride全设置为1。
    # 第1个和第4个必须是1，因为第1个是图像的数目，第4个是图像的通道。
    # 第2和第3指定和左右、上下的步长。
    # padding设置为'SAME' 意味着给图像补零，以保证前后像素相同。
    layer = tf.nn.conv2d(input=input,
                        filter=weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')

    layer += biases

    #layer = tf.abs(tf.nn.tanh(layer))
    layer = tf.nn.relu(layer)

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


        #poolWeights = new_weights(shape=[num_filters])
        #poolBiases = new_biases(length=num_filters)
        #layer = tf.nn.tanh(poolWeights * layer + poolBiases)
        layer = tf.nn.tanh(layer)

    return layer, weights


def flatten_layer(layer):
    # 获取输入层的形状，
    # layer_shape == [num_images, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()

    # 特征数量: img_height * img_width * num_channels
    # 可以使用TensorFlow内建操作计算.
    num_features = layer_shape[1:4].num_elements()

    # 将形状重塑为 [num_images, num_features].
    # 注意只设定了第二个维度的尺寸为num_filters，第一个维度为-1，保证第一个维度num_images不变
    # 展平后的层的形状为:
    # [num_images, img_height * img_width * num_channels]
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input,          # 前一层.
                num_inputs,    # 前一层输入维度
                num_outputs,    # 输出维度
                use_Abs=True): # 是否使用RELU

    # 新的权重和偏置，与第一章一样.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # 计算 y = wx + b，同第一章
    layer = tf.matmul(input, weights) + biases

    # 是否使用RELU
    if use_Abs:
        layer = tf.nn.tanh(layer)

    return layer