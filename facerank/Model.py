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
import os
import pandas as pd
filepath = './data/Data_Collection/'
ratepath = './data/Rating_Collection/allraters.csv'
os.system('ls')
files = os.listdir(filepath)
filelist = [filepath+i for i in files if i.endswith('jpg')]
def handleImg():
    ylabel = pd.read_csv(ratepath)
    #ylabel.dropna()
    print(np.where(np.isnan(ylabel['#image'])))
    ylabel['#image']=ylabel['#image'].apply(lambda x:int(x))
    ylabel1 = ylabel[['#image','Rating']].groupby(['#image']).median()#['Rating']
    #print(ylabel1.loc[1])
    print(ylabel['#image'].head())
    return ylabel1
ylabel = handleImg()
print(ylabel.loc[46]['Rating'])    

def getBatchImage(batch):
    result = []
    yresult = []
    random.shuffle (filelist)
    for i in range(batch):
        if not filelist[i].endswith('jpg'):
            continue
        img = Image.open(filelist[i]).resize((128,128))
        result.append(numpy.array(img)/255)
        num = int(filelist[i].split('-')[-1].split('.')[0])
        #print(num)
        y = int(ylabel.loc[num]['Rating'])-1
        yla = np.zeros(5)
        yla[y]=1
        yresult.append(yla)
        #result.append(img)
    return result,yresult


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

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
lr = 0.01
epoch = 1000
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(lr,
                                           global_step=global_step,
                                           decay_steps=10,decay_rate=0.9)

x = tf.placeholder(tf.float32,[batch_size,128,128,3])
y = tf.placeholder(tf.float32,[batch_size,5])
keep_prob = tf.placeholder(tf.float32)

h1 = makenet(x,[5,5,3,16],[16])    
h2 = makenet(h1,[5,5,16,32],[32])

#full connected
c3 = makefullnet(tf.reshape(h2,[-1,32*32*32]),[32*32*32,1024],[1024],keep_prob)

c4 = makesoftnet(c3,[1024,5],[5])+1e-10

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(c4),reduction_indices=[1]))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(c4,1), tf.argmax(y,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        x_train,y_train=getBatchImage(batch_size)
        global_step=i
        #print(x_train,y_train)
      
        acc,cos,_=sess.run([accuracy,cost,train],feed_dict={x:x_train,y:y_train,keep_prob: 0.8})
        print('epoch=%d,acc=%f,cos=%f'%(i,acc,cos))