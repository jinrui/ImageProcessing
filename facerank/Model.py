#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:04:14 2018

@author: wangquying
"""

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import random

def getBatchImage(filelist,batch):
    result = []
    random.shuffle (filelist)
    for i in range(batch):
        img = Image.open(filelist[i]).resize((128,128))
        result.append(img)
    return img


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape),stddev=0.1)

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))


def conv2d(x,w):
    return tf.nn.conv2d(x,w,[1,1,1,1],padding='SAME')

def maxpool2x2(x):
    return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

def makenet(x,w_shape,b_shape):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    r = tf.nn.relu(conv2d(x,w)+b)
    return maxpool2x2(r)

def makefullnet(x,w_shape,b_shape,keep_prob):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    r = tf.nn.relu(tf.matmul(x,w)+b)
    return tf.nn.dropout(r,keep_prob)

def makesoftnet(x,w_shape,b_shape):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    return tf.nn.softmax(tf.matmul(x,w)+b)

batch_size=16
lr = 0.001
epoch = 100

x = tf.placeholder(tf.float32,[batch_size,128,128,3])
y = tf.placeholder(tf.float32,[batch_size,5])
keep_prob = tf.placeholder(tf.float16)

h1 = makenet(x,[5,5,3,32],[32])    
h2 = makenet(h1,[5,5,32,64],[64])

#full connected
c3 = makefullnet(tf.reshape(h2,[-1,32*32*64]),[32*32*64,1024],[1024],keep_prob)

c4 = makesoftnet(c3,[1024,5],[5])

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(c4)),reduction_indices=1)
train = tf.train.AdamOptimizer(lr).minimize(cost)

correct_prediction = tf.equal(tf.argmax(c4,1), tf.argmax(y,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    for i in range(epoch):
        getBatchImage(batch_size)