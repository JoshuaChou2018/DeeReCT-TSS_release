#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from math import sqrt
import numpy as np
from numpy import zeros
import sys
import re
import math
import os
from tf_model_component import *
from batch_object import batch_object
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pdb
from tensorflow.python.saved_model import builder as saved_model_builder
 
def brun(sess, y, a):
    preds = []
    batch_size = 100
    number_of_full_batch=int(math.ceil(float(len(a))/batch_size))
    for i in range(number_of_full_batch):
        preds += list(sess.run(y, feed_dict={input_x: a[i*batch_size:(i+1)*batch_size], kr1: 1.0, kr2: 1.0}))
    return preds

def encode(s):
    ns = s.upper()
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = ns.replace("A", "0,")
    ns = ns.replace("T", "1,")
    ns = ns.replace("G", "2,")
    ns = ns.replace("C", "3,")
    if re.search('[a-zA-Z]', ns):
        #print(s)
    	print('Non-standard symbol in sequence - changed to A.')
    	ns = re.sub("[a-zA-Z]", "4,", ns)
    return ns[:-1]

np.random.seed(2504) 

#total = len(sys.argv)
#if total<3:
#    print('USAGE: <model> <input file>')
#    exit(0)

#print('\nClassification of promoter and non-promoter sequences\n')
max_features = 4
sLen = int(sys.argv[3])
step = int(sys.argv[4])
output =  str(sys.argv[5])
inp = str(sys.argv[2])
sequences1 = []
names = []
seq = ""
with open(inp) as f:
    for line in f:
        if(line.startswith(">")):
            names.append(line.strip())
            if(len(seq)!=0):
                sequences1.append(np.fromstring(encode(seq), dtype=int, sep=","))
                seq=""                    
            continue                
        else:
            seq+=line

if(len(seq)!=0):
    sequences1.append(np.fromstring(encode(seq), dtype=int, sep=","))

sequences = zeros((len(sequences1),  len(sequences1[0]), max_features))

for i in range(len(sequences1)):
    for j in range(len(sequences[i])):
        if(sequences1[i][j]<4):
            sequences[i][j][sequences1[i][j]]=1

str_list = []
new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    # Import the previously export meta graph.
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], sys.argv[1])
    # Restore the variables
    saver = tf.train.Saver()
    saver.restore(sess, sys.argv[1]+"/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr1 = tf.get_default_graph().get_tensor_by_name("kr1:0")
    kr2 = tf.get_default_graph().get_tensor_by_name("kr2:0")
    for i in range(len(sequences)):
        total = int(math.ceil((len(sequences[i]) - sLen) / step) + 1);
        topred = np.zeros(shape=(total,sLen,max_features))
        for j in range(total):
            topred[j] = sequences[i][j * step : j * step + sLen]
        predict = brun(sess, y, topred)
        str_list.append("Sequence "+ str(i) + "\n")
        prefix = ""
        for j in range(total):
            str_list.append(prefix)
            score = (predict[j][0] - predict[j][1] + 1.0)/2.0
            #print(str(predict[j][0]) + "  ---  " + str(predict[j][1]))
            str_list.append(str(score))
            prefix = ", "
        str_list.append("\n")
        if(i%100==0):
        	print(str(i)) 

with open(output, 'w+') as f:
    f.write(''.join(str_list))





