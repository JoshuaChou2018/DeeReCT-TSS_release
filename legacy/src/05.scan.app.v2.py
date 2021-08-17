# python=3.6
import tensorflow as tf
from pathlib import Path
from random import shuffle
import math
import os
import pickle
import numpy as np
from numpy import zeros
from math import sqrt
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def gen_hyper_dict(hyper_dict=None):
    def rand_log(a, b):
        x = np.random.sample()
        return 10.0 ** ((np.log10(b) - np.log10(a)) * x + np.log10(a))

    def rand_sqrt(a, b):
        x = np.random.sample()
        return (b - a) * np.sqrt(x) + a

    hyper_dict = {
        'tf_learning_rate': rand_log(.0005, .05),
        'tf_momentum': rand_sqrt(.95, .99),
        'tf_motif_init_weight': rand_log(1e-2, 10),
        'tf_fc_init_weight': rand_log(1e-2, 10),
        'tf_motif_weight_decay': rand_log(1e-5, 1e-3),
        'tf_fc_weight_decay': rand_log(1e-5, 1e-3),
        'tf_keep_prob': np.random.choice([.5, .75, 1.0]),
        #'tf_keep_prob': 1.0,
    }
    return hyper_dict

## Model

class Net:
    def __init__(self,hyper_dict=None):
        self.hyper_dict=hyper_dict
        self.build()

    def build(self):
        sequence_length = 1001
        num_channels = 4
        num_classes = 2
        patch_size = 10
        num_depth = 64  # default 16, 64 works good
        num_seq_hidden1 = 128  # default 64, 128 works good
        num_cov_hidden1 = 128  # default 64, 128 works good
        num_groups = 4
        filter_size = 10
        maxpool_size = 10

        # Hyperdict
        if self.hyper_dict==None:
            self.hyper_dict=gen_hyper_dict()
        else:
            self.hyper_dict=self.hyper_dict

        for key in self.hyper_dict.keys():
            _info = '[model]\t{}: {}'.format(key,self.hyper_dict[key])
            print(_info)

        tf_motif_init_weight = self.hyper_dict['tf_motif_init_weight']
        tf_fc_init_weight = self.hyper_dict['tf_fc_init_weight']
        tf_motif_weight_decay = self.hyper_dict['tf_motif_weight_decay']
        tf_fc_weight_decay = self.hyper_dict['tf_fc_weight_decay']
        tf_keep_prob = self.hyper_dict['tf_keep_prob']

        # Share
        self.istrain = tf.placeholder(tf.bool, shape=[])
        self.label_data = tf.placeholder(tf.float32, shape=(None, num_classes))

        # Sequence Model
        self.seq_data = tf.placeholder(tf.float32, shape=(None, sequence_length, 1, num_channels))
        print('seq_data shape ', self.seq_data.shape)
        seq_convolution1_weights = tf.Variable(
            tf.truncated_normal([patch_size, 1, num_channels, num_depth], stddev=tf_motif_init_weight))
        print('seq_convolution1_weights shape ', seq_convolution1_weights.shape)
        seq_convolution1_biases = tf.Variable(tf.zeros([num_depth]))
        print('seq_convolution1_biases shape ', seq_convolution1_biases.shape)
        seq_hidden1_weights = tf.Variable(
            tf.truncated_normal([(sequence_length - patch_size + 1) // filter_size * num_depth, num_seq_hidden1],
                                stddev=tf_fc_init_weight))
        print('seq_hidden1_weights shape ', seq_hidden1_weights.shape)
        seq_hidden1_biases = tf.Variable(tf.constant(1.0, shape=[num_seq_hidden1]))
        print('seq_hidden1_biases shape ', seq_hidden1_biases.shape)

        seq_convolution1 = tf.nn.conv2d(self.seq_data, seq_convolution1_weights, [1, 1, 1, 1], padding='VALID')
        print('seq_convolution1 shape ', seq_convolution1.shape)
        seq_convolution1 = tf.reshape(seq_convolution1, [-1, sequence_length - patch_size + 1, 1, num_depth])
        print('seq_convolution1 shape ', seq_convolution1.shape)
        seq_hidden1 = tf.nn.relu(seq_convolution1 + seq_convolution1_biases)
        print('seq_hidden1 shape ', seq_hidden1.shape)
        seq_hidden1 = tf.nn.max_pool(seq_hidden1, [1, maxpool_size, 1, 1], [1, maxpool_size, 1, 1], padding='VALID')
        print('seq_hidden1 shape ', seq_hidden1.shape)
        seq_hidden1_shape = seq_hidden1.get_shape().as_list()
        seq_motif_score = tf.reshape(seq_hidden1, [-1, seq_hidden1_shape[1] * num_depth])
        print('seq_motif_score shape ', seq_motif_score.shape)
        seq_hidden2 = tf.nn.dropout(tf.nn.relu(tf.matmul(seq_motif_score, seq_hidden1_weights) + seq_hidden1_biases),
                                    tf_keep_prob)
        print('seq_hidden2 shape ', seq_hidden2.shape)

        # Coverage Model
        self.cov_data = tf.placeholder(tf.float32, shape=(None, sequence_length, 1, 1))
        cov_convolution1_weights = tf.Variable(
            tf.truncated_normal([patch_size, 1, 1, num_depth], stddev=tf_motif_init_weight))
        cov_convolution1_biases = tf.Variable(tf.zeros([num_depth]))
        cov_hidden1_weights = tf.Variable(
            tf.truncated_normal([(sequence_length - patch_size + 1) // filter_size * num_depth, num_cov_hidden1],
                                stddev=tf_fc_init_weight))
        cov_hidden1_biases = tf.Variable(tf.constant(1.0, shape=[num_cov_hidden1]))

        cov_convolution1 = tf.nn.conv2d(self.cov_data, cov_convolution1_weights, [1, 1, 1, 1], padding='VALID')
        cov_convolution1 = tf.reshape(cov_convolution1, [-1, sequence_length - patch_size + 1, 1, num_depth])
        cov_hidden1 = tf.nn.relu(cov_convolution1 + cov_convolution1_biases)
        cov_hidden1 = tf.nn.max_pool(cov_hidden1, [1, maxpool_size, 1, 1], [1, maxpool_size, 1, 1], padding='VALID')
        cov_hidden1_shape = cov_hidden1.get_shape().as_list()
        cov_motif_score = tf.reshape(cov_hidden1, [-1, cov_hidden1_shape[1] * num_depth])
        cov_hidden2 = tf.nn.dropout(tf.nn.relu(tf.matmul(cov_motif_score, cov_hidden1_weights) + cov_hidden1_biases),
                                    tf_keep_prob)

        # Merge feature map
        merge_weights = tf.Variable(tf.truncated_normal(
            [num_cov_hidden1 + num_seq_hidden1, num_classes], stddev=tf_fc_init_weight))
        merge_biases = tf.Variable(tf.constant(1.0, shape=[num_classes]))

        merged_hidden_nodes = tf.concat([seq_hidden2, cov_hidden2], 1)
        logits = tf.matmul(merged_hidden_nodes, merge_weights) + merge_biases
        self.logits = logits
        self.prediction = tf.nn.softmax(logits)

        # save weight
        weights = {}
        weights['seq_convolution1_weights'] = seq_convolution1_weights
        weights['seq_convolution1_biases'] = seq_convolution1_biases
        weights['seq_hidden1_weights'] = seq_hidden1_weights
        weights['seq_hidden1_biases'] = seq_hidden1_biases

        weights['cov_convolution1_weights'] = cov_convolution1_weights
        weights['cov_convolution1_biases'] = cov_convolution1_biases
        weights['cov_hidden1_weights'] = cov_hidden1_weights
        weights['cov_hidden1_biases'] = cov_hidden1_biases

        weights['merge_weights'] = merge_weights
        weights['merge_biases'] = merge_biases
        self.weights = weights

    def build_train_graph(self):
        hd = self.hyper_dict
        wts = self.weights
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_data,logits=self.logits))
        #self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_data, logits=self.logits))

        # Optimizer.
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        self.stepOp = tf.assign_add(global_step, 1).op
        learning_rate = tf.train.exponential_decay(hd['tf_learning_rate'], global_step, 3000, 0.96)
        self.optimizeOp = tf.train.MomentumOptimizer(learning_rate, hd['tf_momentum']).minimize(self.loss)

    def predict(self, sess, seq_data, cov_data, istrain=False):
        fd = {self.seq_data: seq_data, self.cov_data: cov_data, self.istrain: istrain}
        return sess.run(self.prediction, feed_dict=fd)

    def load_weights(self, wts, sess):
        wts = np.load(wts)
        ph = tf.placeholder(tf.float32)
        for k in self.weights:
            sess.run(tf.assign(self.weights[k], ph).op, feed_dict={ph: wts[k]})


def Performance(predict, y_test):
    a = zeros(len(y_test))
    for i in range(len(predict)):
        if predict[i][0] > 0.5:
            a[i] = 1
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    tp_idx = []
    tn_idx = []
    fp_idx = []
    fn_idx = []

    for i in range(len(y_test)):
        if (y_test[i][0] == 1):
            if (a[i] == 1):
                tp += 1
                tp_idx.append(i)
            else:
                fn += 1
                fn_idx.append(i)
        if (y_test[i][0] == 0):
            if (a[i] == 1):
                fp += 1
                fp_idx.append(i)
            else:
                tn += 1
                tn_idx.append(i)
    sn = 0.0
    sp = 0.0
    mcc = 0.0

    tpr, tnr, fpr, acc, mcc, fdr, f1 = -1, -1, -1, -1, -1, -1, -1
    try:
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        fdr = fp / (fp + tp)
        f1 = 2 * tp / (2 * tp + fp + fn)
    except Exception:
        pass
    return tp, tn, fp, fn, tpr, tnr, fpr, acc, mcc, fdr, f1, tp_idx, tn_idx, fp_idx, fn_idx


def idx2info(tp_idx, tn_idx, fp_idx, fn_idx, data):
    tp_info = []
    tn_info = []
    fp_info = []
    fn_info = []
    for idx in tp_idx:
        tp_info.append(data[idx])
    for idx in tn_idx:
        tn_info.append(data[idx])
    for idx in fp_idx:
        fp_info.append(data[idx])
    for idx in fn_idx:
        fn_info.append(data[idx])
    return tp_info, tn_info, fp_info, fn_info


def load_model(load_model_path, hyper_dict, rep_id=0):

    # with tf.device('/device:GPU:0'):
    if True:

        model = Net(hyper_dict=hyper_dict)
        model.build_train_graph()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        model.load_weights(load_model_path, sess)

    return sess, model


def load_rnaseq_coverage_data(rnaseq_file_path_posi,rnaseq_file_path_neg):
    data = {'+':{},'-':{}}
    try:
        with open(rnaseq_file_path_posi, 'r') as f:
            for line in f:
                try:
                    line = line.rstrip().split('\t')
                    chr = line[0]
                    pos = int(line[1])
                    count = int(float(line[2]))
                    try:
                        data['+'][chr][pos] = count
                    except:
                        data['+'][chr] = {}
                        data['+'][chr][pos] = count
                except:
                    print(line)
    except:
        print('[ERROR] positive depth nor prodived!')
    try:
        with open(rnaseq_file_path_neg, 'r') as f:
            for line in f:
                line = line.rstrip().split('\t')
                chr = line[0]
                pos = int(line[1])
                count = int(float(line[2]))
                try:
                    data['-'][chr][pos] = count
                except:
                    data['-'][chr] = {}
                    data['-'][chr][pos] = count
    except:
        print('[ERROR] negative depth nor prodived!')
    return data

def load_ref_data(ref_file_path):
    data = {}
    datas = SeqIO.parse(ref_file_path, format='fasta')
    for record in datas:
        data[record.id.split(' ')[0]] = record.seq
    return data

def scan_load_basic_data():
    global rnaseq_coverage_data
    global ref_data

    _info = '[info]\tload basic data for scan'
    print(_info)

    # load rnaseq data
    try:
        rnaseq_coverage_data = pickle.load(open(scanbed + '.coverage_data.p', 'rb'))
        print('loaded rnaseq coverage data from saved data')
    except:
        rnaseq_coverage_data = load_rnaseq_coverage_data(rnaseq_file_path_posi, rnaseq_file_path_neg)
        pickle.dump(rnaseq_coverage_data, open(scanbed + '.coverage_data.p', 'wb'))
        print('loaded rnaseq coverage data from raw depth and saved to rnaseq_coverage_data.p')

    # load fasta data
    try:
        ref_data = pickle.load(open(root_path + '/ref_data.p', 'rb'))
        print('loaded ref data from saved data')
    except:
        ref_data = load_ref_data(ref_file_path)
        pickle.dump(ref_data, open(root_path + '/ref_data.p', 'wb'))
        print('loaded ref data from hg.fasta and saved to ref_data.p')

    _info = '[info]\tfinish load basic data for scan'
    print(_info)

def minmax(data):
    tmp = []
    if max(data) - min(data) == 0:
        for x in data:
            tmp.append(0)
    else:
        for x in data:
            tmp.append((x - min(data)) / (max(data) - min(data)))
    return tmp

def scan_load_data(_chr='chr1', scan_start=1, scan_end=10001, pos=100, strand='+',mm=False,step_size=1):
    # scan_halfsize = 5000
    # scan_start = scan
    # scan_end = pos + scan_halfsize + 1

    valid_data = []
    valid_cage_cov = []
    valid_rnaseq_cov = []
    scan_center=pos
    for center in range(scan_start, scan_end,step_size):
        start = center - half_size
        end = center + half_size + 1
        coverage = []
        try:
            valid_cage_cov.append(0)
        except:
            valid_cage_cov.append(0)
        try:
            valid_rnaseq_cov.append(rnaseq_coverage_data[strand][_chr][center])
        except:
            valid_rnaseq_cov.append(0)
        for x in range(start, end):
            try:
                coverage.append(rnaseq_coverage_data[strand][_chr][x])
            except:
                coverage.append(0)
        if mm == True:
            coverage = minmax(coverage)

        if np.abs(scan_center-center)<=tolerant_distance: # distance to center of tss, then positive
            tmp = ['{}_{}_{}_{}'.format(_chr, start, end, strand),
                   {'chr': _chr, 'start': start, 'end': end, 'score': 1, 'strand': strand},
                   str(ref_data[_chr][start:end].upper()), coverage]
        else: # distance to center of tss >100, then negative
            tmp = ['{}_{}_{}_{}'.format(_chr, start, end, strand),
                   {'chr': _chr, 'start': start, 'end': end, 'score': -1, 'strand': strand},
                   str(ref_data[_chr][start:end].upper()), coverage]
        valid_data.append(tmp)

    def split_seq_cov(valid_data):
        valid_info = []
        valid_seq = []
        valid_cov = []
        valid_label = []

        def encode_seq(seq, strand):
            if strand == "+":
                rep = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1], "N": [0, 0, 0, 0]}
            else:
                seq = seq[::-1]
                rep = {"A": [0, 1, 0, 0], "T": [1, 0, 0, 0], "C": [0, 0, 0, 1], "G": [0, 0, 1, 0], "N": [0, 0, 0, 0]}
            tmp = []
            for x in seq:
                tmp.append(rep[x])
            return tmp

        def encode_cov(cov, strand):
            if strand == '+':
                return np.array(cov)
            elif strand == '-':
                return cov[::-1]

        for data in valid_data:
            info = data[1]
            seq = data[2]
            cov = data[3]
            strand = info['strand']
            # print(len(seq),len(cov),sequence_length)
            if len(seq) == len(cov) == sequence_length:
                if info['score'] == -1:
                    label = [0, 1]  # not tss
                else:
                    label = [1, 0]  # real tss
                valid_info.append(info)
                valid_seq.append(encode_seq(seq, strand))
                valid_cov.append(encode_cov(cov, strand))
                valid_label.append(label)
            else:
                print('error', info)

        return valid_info, valid_seq, valid_cov, valid_label

    valid_info, valid_seq, valid_cov, valid_label = split_seq_cov(valid_data)

    valid_seq = np.array(valid_seq).reshape(-1, sequence_length, 1, 4)
    valid_cov = np.array(valid_cov).reshape(-1, sequence_length, 1, 1)
    valid_real_seq = str(ref_data[_chr][scan_start:scan_end].upper())

    return valid_data, valid_info, valid_seq, valid_cov, valid_label, valid_real_seq, valid_cage_cov, valid_rnaseq_cov

def scan(sess, model, _chr='chr1', scan_start=1, scan_end=10001, pos=100, strand='+',mm=False,reducenoise=False,step_size=1):
   
    global output_file
    global cutoff

    _info = '[info]\tstart scanning {}:{}...{};{};{}'.format(_chr, scan_start, scan_end, strand, pos)
    print(_info)

    ### load data
    scan_data, scan_info, scan_seq, scan_cov, scan_label, scan_real_seq, scan_cage_cov, scan_rnaseq_cov = scan_load_data(
        _chr, scan_start, scan_end, pos, strand,mm=mm,step_size=step_size)

    ### start scan

    total_number = int(math.ceil(float(len(scan_label)) / num_batch_size))
    prediction_scan = []

    for i in range(total_number):
        batch_seq_data = scan_seq[i * num_batch_size:(i + 1) * num_batch_size]
        batch_cov_data = scan_cov[i * num_batch_size:(i + 1) * num_batch_size]
        batch_labels = scan_label[i * num_batch_size:(i + 1) * num_batch_size]
        prediction_scan += list(model.predict(sess, batch_seq_data, batch_cov_data, False))

    data=[x[0] for x in prediction_scan]
    for i in range(len(data)):
        if data[i]>=cutoff:
            output_file.write('{}\t{}\t{}\t{:.4f}\t{}\n'.format(_chr,scan_start+i*step_size,scan_start+i*step_size,data[i],strand))

    _info = '[info]\tfinish scan'
    print(_info)

    return None

if __name__ == '__main__':

        # SETTING

        scanbed = sys.argv[1] # 'path/to/scan.bed'
        root_path = sys.argv[2] #'path/to/data/root'
        load_model_path = sys.argv[3] #'path/to/model.npz'
        rnaseq_file_path_posi = sys.argv[4]
        rnaseq_file_path_neg = sys.argv[5]
        ref_file_path = sys.argv[6] #'ref/hg19/hg19.fa'
        try:
            step_size=int(sys.argv[7])
        except:
            step_size=1
        try:
            use_GPU=int(sys.argv[8])
        except:
            use_GPU=1
        try:
            cutoff=float(sys.argv[9])
        except:
            cutoff=0.1
        
        
        if use_GPU==0:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"

        # 20000000 positions ~ 120G MEM
        # 3000 batch size ~ 6G GPU

        num_batch_size = 8000 # 8452 max positions for 12G
        sequence_length = 1001
        half_size = 500
        tolerant_distance=100
        mm=False
        reducenoise = True

        # this hyper dict is very important
        hyper_dict = {
                'tf_learning_rate': 0.0001,
                'tf_momentum': 0.98,
                'tf_motif_init_weight': 0.12,
                'tf_fc_init_weight': 0.02,
                'tf_motif_weight_decay': 0.0004,
                'tf_fc_weight_decay': 0.0003,
                'tf_keep_prob': 0.5,
            }
        #hyper_dict=None

        ### load basic data for scan
        scan_load_basic_data()

        sess, model = load_model(load_model_path, hyper_dict=hyper_dict)

        with open('{}.bedgraph'.format(scanbed),'w') as output_file:
            with open(scanbed,'r') as f:
                scancount=0
                for line in f:
                    print(scancount)
                    #if scancount==200:
                    #    break
                    scancount+=1
                    line=line.rstrip().split('\t')
                    _chr = line[0]
                    scan_start = int(line[1])
                    scan_end = int(line[2])
                    pos=(scan_start+scan_end)/2
                    strand = line[5]
                    
                    try:
                        scan(sess, model, _chr=_chr, scan_start=scan_start, scan_end=scan_end, pos=pos, strand=strand,
                                     mm=mm, reducenoise=reducenoise,step_size=step_size)
                    except:
                        print('*error*' * 10)
