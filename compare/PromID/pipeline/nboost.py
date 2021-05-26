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
import csv
import random

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
    	#print('Non-standard symbol in sequence - changed to A.')
    	ns = re.sub("[a-zA-Z]", "4,", ns)
    return ns[:-1]

def fastarev(a):
    sb = []
    for i in range(len(a)):
        if (a[i][0] == 1):
            sb.append("A")
        elif (a[i][1] == 1):
            sb.append("T")
        elif (a[i][2] == 1):
            sb.append("G")
        elif (a[i][3] == 1):
            sb.append("C")
        else:
            sb.append("N")            
    return ''.join(sb)

def notclose(d, a, step):
    r = True
    for i in range(len(a)):
        if(abs(a[i] - d) < 500/step):
            r = False
            break
    return r

np.random.seed(2504) 

#total = len(sys.argv)
#if total<3:
#    print('USAGE: <model> <input file>')
#    exit(0)

#print('\nClassification of promoter and non-promoter sequences\n')
max_features = 4
sLen = int(sys.argv[3])
step = int(sys.argv[4])
pos = int(sys.argv[5])
d = int(sys.argv[6])
mps = int(sys.argv[7])
output =  str(sys.argv[8])
maxNeg =  int(sys.argv[9])
inp = str(sys.argv[2])
DIR = os.path.dirname(os.path.abspath(sys.argv[2]))
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

#np.random.shuffle(sequences)
#sequences = sequences[:5000]
chosen = []
try:
    with open( DIR+"/"+"chosen", "r") as ins:
        for line in ins:
            if line.strip():     
                chosen.append([int(el) for el in line.split(',')])
            else:
                chosen.append([])
except Exception as e:
    print("no previous file")
    chosen = [[] for x in range(len(sequences))]


str_list = []
cc=0
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
    rol = list(range(len(sequences)))
    random.shuffle(rol)
    iii = 0
    for i in rol:
        if(len(chosen[i]) > 4):
            continue
        iii = iii + 1
        total = int(math.ceil((len(sequences[i]) - sLen) / step) + 1)
        topred = np.zeros(shape=(total,sLen,max_features))
        for j in range(total):
            topred[j] = sequences[i][j * step : j * step + sLen]
        predict = brun(sess, y, topred)     
        scores = np.zeros(shape=(total))
        for j in range(total):
            scores[j] = (predict[j][0] - predict[j][1] + 1.0)/2.0
        inds = np.argsort(scores)
        scores = scores[inds]
        newChosen = 0
        for k, e in reversed(list(enumerate(scores))):
            if(e>0.5):
                if(inds[k]  < (pos - d) / step or inds[k]   > (pos + d) / step):
                    if(notclose(inds[k], chosen[i], step)):
                        str_list.append(">Sequence " + "\n")
                        str_list.append(fastarev(topred[inds[k]]))
                        str_list.append("\n")
                        chosen[i].append(inds[k])
                        cc=cc+1
                        newChosen = newChosen + 1
            else:
                break
            if(newChosen>=mps):
                break
            
        if(iii%100==0):
            print(str(iii) + " - (" + str(cc)+")") 
        if(cc>maxNeg):
            break

with open(output, 'w+') as f:
    f.write(''.join(str_list))

with open( DIR+"/"+'chosen', 'w+') as f:
    writer = csv.writer(f)
    writer.writerows(chosen)