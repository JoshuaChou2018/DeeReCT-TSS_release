#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from math import sqrt
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
import random
import time
import shutil
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
weight_decay = 5e-4
learning_rate = 0.0001
output_step = 10

def conv1d(input, name, kshape, stride=1):
    with tf.name_scope(name):
        W = tf.get_variable(name='fre_w_'+name,
                            shape=kshape)
        b = tf.get_variable(name='fre_bias_' + name,
                            shape=[kshape[2]])
        out = tf.nn.conv1d(input,W,stride=stride, padding='VALID')
        out = tf.nn.bias_add(out, b)
        out = tf.maximum(out, 0.001 * out)
        return out

def fullyConnected(input, name, output_size):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='fre_w_'+name,
                            shape=[input_size, output_size])
        b = tf.get_variable(name='fre_bias_'+name,
                            shape=[output_size])
        input = tf.reshape(input, [-1, input_size])
        out = tf.add(tf.matmul(input, W), b)
        out = tf.maximum(out, 0.001 * out, name = "forsoft")
        return out

# define the graph
def model_graph(x, keep_ratio1, keep_ratio2):
    dpi = tf.nn.dropout(x, keep_ratio1)  
    f1 = conv1d(dpi, name='f1', kshape=[15, 4, 32])
    f2 = conv1d(f1, name='f2', kshape=[15, 32, 64])
    #f2 = tf.layers.max_pooling1d(f2, 15, 1, padding='VALID')
    f3 = tf.contrib.layers.flatten(tf.nn.dropout(f2, keep_ratio2))

    a1 = conv1d(x, name='a1', kshape=[1, 4, 4])
    af = tf.contrib.layers.flatten(tf.layers.average_pooling1d(a1, 15, 15, padding='VALID'))


    #fc = tf.contrib.layers.flatten(f2)
    fc = tf.concat([f3, af], 1)
    #print("-----------------------------------------------------------------------")
    #print(fc)
    #fc = fullyConnected(f2, "fc", 256)
    wn = 36768
    #wn = 256
    #print("weights for softmax1: " + str(wn))
    #dp = tf.nn.dropout(fc, keep_ratio2)  
    with tf.name_scope('softmax_layer'):
        W = tf.get_variable(name = "W_soft", shape=[wn, 2])
        b = tf.get_variable(name = "bias_soft", shape=[2])
        y_logit = tf.add(tf.matmul(fc, W), b)
        s_out = tf.nn.softmax(y_logit)

    return y_logit, s_out

def testMCC(predict, y_test):
    a = zeros(len(y_test))
    for i in range(len(predict)):
        if predict[i][0] - predict[i][1] > 0:
            a[i] = 1
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0

    for i in range(len(y_test)):
        if(y_test[i][0] == 1):
            if(a[i] == 1):
                tp+=1
            else:
                fn+=1
        if(y_test[i][0] == 0):
            if(a[i] == 1):
                fp+=1
            else:
                tn+=1
    sn = 0.0
    sp = 0.0
    mcc = 0.0
    try:
        sn = tp/(tp+fn)
        sp = tn/(tn+fp)
        mcc = (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    except Exception: 
        pass
    return (tp, tn, fp, fn, sn, sp, mcc)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

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

np.random.seed(2504) 

p = {'maxlen':81, 'batch_size':16, 'hidden_dims':128, 'nb_epoch':5,
'pos_seq':'post.fa', 'neg_seq':'negt.fa', 'model_output':'model.h5', 'learning_split':0.6, 'mix_rate':0.0,'mix_amount':0.0,
'testing_split':0.3,'validation_split':0.1,'additional_negatives':0.0,
 'patience':10,  'read_model': 0, 'test_results_file':'results', 'threshold':0, 'kr1':1.0,'kr2':1.0, 'cw1':1.0, 'model_file': 'model.h5',
   'architecture':[[200, 21, 2]]  }
with open(sys.argv[1]) as f:
    for line in f:
        if(not line.strip().startswith("#") and len(line)>0):
            a = line.split("=")
            k = a[0].strip()
            v = a[1].strip()
            if k in p:
                if k=='architecture':
                    p[k]= np.fromstring(v, dtype=int, sep=',').reshape((-1, 3))
                elif v.isdigit():
                    p[k] = int(v)
                elif is_number(v):
                    p[k] = float(v)
                else:
                    p[k] = v
DIR = os.path.dirname(os.path.abspath(sys.argv[2]))


p['model_output'] = str(sys.argv[2])

cw1 = 1.0
cw2 = 1.0
max_features = 4  
pos_seq = []
seq = ""
with open(p['pos_seq']) as f:
    for line in f:
        if(line.startswith(">")):
            if(len(seq)!=0):
                pos_seq.append(np.fromstring(encode(seq), dtype=int, sep=","))
                seq=""                    
            continue                
        else:
            seq+=line
if(len(seq)!=0):
    pos_seq.append(np.fromstring(encode(seq), dtype=int, sep=","))

neg_seq = []
seq = ""
with open(p['neg_seq']) as f:
    for line in f:
        if(line.startswith(">")):
            if(len(seq)!=0):
                neg_seq.append(np.fromstring(encode(seq), dtype=int, sep=","))
                seq=""                    
            continue                
        else:
            seq+=line
if(len(seq)!=0):
    neg_seq.append(np.fromstring(encode(seq), dtype=int, sep=","))

np.random.shuffle(pos_seq)
np.random.shuffle(neg_seq)
#neg_seq = neg_seq[:len(pos_seq)]


lpn = int(math.floor((p['learning_split'] + p['validation_split'])*len(pos_seq)))
lnn = int(math.floor((p['learning_split'] + p['validation_split'])*len(neg_seq)))
tpn = int(math.ceil(p['testing_split']*len(pos_seq)))
tnn = int(math.ceil(p['testing_split']*len(neg_seq)))

sl = len(pos_seq[0])
x_data = zeros((len(pos_seq) + len(neg_seq),  sl, max_features))
y_data = zeros((len(pos_seq) + len(neg_seq), 2))

for i in range(len(pos_seq)):
    y_data[i] = (1, 0)
    for j in range(sl):
        if(pos_seq[i][j]<4):
            x_data[i][j][pos_seq[i][j]]=1

for i in range(len(neg_seq)):
    y_data[len(pos_seq) + i] = (0, 1)
    for j in range(sl):
        if(neg_seq[i][j]<4):
            x_data[len(pos_seq) + i][j][neg_seq[i][j]]=1

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2504)

print('Parameters File: ' + sys.argv[1])
print('Positives: ' + p['pos_seq'] + '   ' + str(len(pos_seq)))
print('Negatives: ' + p['neg_seq'] + '   ' + str(len(neg_seq)))
print('Positives for training: ' + str(lpn))
print('Negatives for training: ' + str(lnn))
print('Positives for testing:  ' + str(tpn))
print('Negatives for testing:  ' + str(tnn))

#cw1=4.0

# define the optimizer and so on and do a random initialization of the graph
batch_size = p['batch_size']
nb_epoch=p['nb_epoch']
input_x = tf.placeholder(tf.float32,shape=[None, np.shape(x_train)[1], 
    max_features], name='input_prom')
y_ = tf.placeholder(tf.float32, [None, 2])
kr1 = tf.placeholder(tf.float32, name="kr1")
kr2 = tf.placeholder(tf.float32, name="kr2")
y_logit, y = model_graph(input_x, kr1, kr2)
tf.identity(y, name="output_prom")
# Define loss and optimizer
weight_array = np.array([cw1, cw2])
weight_per_data = tf.transpose( tf.matmul(y_, 
    tf.reshape(tf.cast(tf.constant(weight_array), tf.float32),
    [len(weight_array),1])))
cost = tf.reduce_mean(tf.multiply(weight_per_data,
    tf.nn.softmax_cross_entropy_with_logits(
    logits=y_logit, labels=y_)))
vars   = tf.trainable_variables() 
l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ])
cost = tf.add(cost, weight_decay*l2_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Start the session
sess = tf.InteractiveSession(config=tf.ConfigProto(
    log_device_placement=False,allow_soft_placement=True))
# Initializing the variables
sess.run(tf.global_variables_initializer())
bmcc = -1
wait = 0
saver = tf.train.Saver()
for epoch in range(nb_epoch):
    #x_train_obj = batch_object(x_train, batch_size)
    #y_train_obj = batch_object(y_train, batch_size)
    for step in range(int(len(x_train)/batch_size)+1):
        #x_train_batch = x_train_obj.next_batch()
        #y_train_batch = y_train_obj.next_batch()
        train_batch = generate_random_batch([x_train], y_train, batch_size)
        sess.run(optimizer, feed_dict={
            input_x: list(train_batch[0])[0],
            y_: train_batch[1], 
            kr1: p['kr1'],
            kr2: p['kr2']
            })      
    print('Result after training epoch {}'.format(epoch))
    pred = []
    number_of_full_batch=int(math.ceil(float(len(x_train))/batch_size))
    for i in range(number_of_full_batch):
        pred += list(sess.run(y, feed_dict={input_x: x_train[i*batch_size:(i+1)*batch_size], kr1: 1.0, kr2: 1.0}))
    pred = np.asarray(pred)
    tp, tn, fp, fn, sn, sp, mcc = testMCC(pred, y_train)
    print("Training set mcc, sn, sp: " + str(mcc) + ", "+str(sn) + ", " + str(sp))   
    
    pred = []
    number_of_full_batch=int(math.ceil(float(len(x_val))/batch_size))
    for i in range(number_of_full_batch):
        pred += list(sess.run(y, feed_dict={input_x: x_val[i*batch_size:(i+1)*batch_size], kr1: 1.0, kr2: 1.0}))
    pred = np.asarray(pred)
    tp, tn, fp, fn, sn, sp, mcc = testMCC(pred, y_val)
    print("Validation set mcc, sn, sp: " + str(mcc) + ", "+str(sn) + ", " + str(sp))   
    if(mcc > bmcc):
        bmcc = mcc
        wait = 0  
        saver.save(sess, DIR+"/" + "model_temp")
    else:
        wait = wait + 1
    if(wait >= p['patience'] ):
        break

# define the model storage part
saver.restore(sess, "model_temp")

pred = []
number_of_full_batch=int(math.ceil(float(len(x_val))/batch_size))
for i in range(number_of_full_batch):
    pred += list(sess.run(y, feed_dict={input_x: x_val[i*batch_size:(i+1)*batch_size], kr1: 1.0, kr2: 1.0}))
pred = np.asarray(pred)
tp, tn, fp, fn, sn, sp, mcc = testMCC(pred, y_val)
print("Final mcc, sn, sp: " + str(mcc) + ", "+str(sn) + ", " + str(sp))   

out_dir = DIR+"/"+p['model_output']
if(os.path.exists(out_dir)):
      out_dir = out_dir + str(time.time())
builder = tf.saved_model.builder.SavedModelBuilder(out_dir)
predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(input_x)
predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(y)
prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs={"input_prom": predict_tensor_inputs_info}, outputs={"output_prom": predict_tensor_scores_info}, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={"model": prediction_signature})
builder.save(True)