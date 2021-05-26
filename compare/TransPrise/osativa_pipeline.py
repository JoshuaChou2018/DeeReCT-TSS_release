"""
Pipeline for training and testing CNN model for prediction TSS with dna_seq data from Oriza Sativa
"""
import dataset
import models
import metrics
import numpy as np
import time

t0 = time.time()

# !!! 1. Dataset from dna-seq data and TSS list file (Chromosome, Strand, Locus_ID, TSS_position)
# !!! 1.1 Read DNA-seq data and TSS positions file
TSS = dataset.tss_read('./data/TSS_MSU.txt', sep=',')  # read TSS_MSU file
split = dataset.split_data(TSS)  # split given chromosomes for train and test sets


dataset.sequences(tss_array=TSS, dna_seq='./data/all.con', split=split, new_path='./data/', nucleotides=1000)
# read all.con file (DNA-seq data) and save 4 new files: train_seq.fa, train_tss_pos.fa, test_seq.fa, test_tss_pos.fa


dataset.beyond_genes(tss_array=TSS, dna_seq='./data/all.con', split=split, examples_train=15000, examples_test=1000,
                     new_path='./data/', nucleotides=512, distance=2000)
# read all.con file (DNA_seq data) and save 2 new files with 10k examples of beyond genes space sequences:
# train_nogenes_seqs.fa, test_nogenes_seqs.fa

t1 = time.time()
# !!! 1.2 Dataset for classification
train_set, train_answers = dataset.class_assemble(sequences='./data/train_sequences.fa',
                                                  beyond_genes_seqs='./data/train_nogenes_seqs.fa',
                                                  tss_pos='./data/train_tss_pos.fa', examples=60000, length=512,
                                                  min_pos=250, max_pos=450)

test_set, test_answers = dataset.class_assemble(sequences='./data/test_sequences.fa',
                                                beyond_genes_seqs='./data/test_nogenes_seqs.fa',
                                                tss_pos='./data/test_tss_pos.fa', examples=4000, length=512,
                                                min_pos=250, max_pos=450)

np.save('./data/train_class_set', train_set)
np.save('./data/train_class_answers', train_answers)
np.save('./data/test_class_set', test_set)
np.save('./data/test_class_answers', test_answers)

t2 = time.time()
# !!! 1.3 Dataset for regression
train_set, train_answers = dataset.regr_assemble(sequences='./data/train_sequences.fa',
                                                 tss_pos='./data/train_tss_pos.fa', examples=40000, length=512,
                                                 min_pos=250, max_pos=450)

test_set, test_answers = dataset.regr_assemble(sequences='./data/test_sequences.fa',
                                               tss_pos='./data/test_tss_pos.fa', examples=1000, length=512,
                                               min_pos=250, max_pos=450)

np.save('./data/train_regr_set', train_set)
np.save('./data/train_regr_answers', train_answers)
np.save('./data/test_regr_set', test_set)
np.save('./data/test_regr_answers', test_answers)

t3 = time.time()
# !!! 2. Models train
# !!! 2.1 Classification model

print('Training classification model')

train_set = np.load('./data/train_class_set.npy')
train_answers = np.load('./data/train_class_answers.npy')

model = models.classification_model(train_set[0].shape)
model.fit(train_set, train_answers, batch_size=64, epochs=5, verbose=2, validation_split=0.1)

test_set = np.load('./data/test_class_set.npy')
test_answers = np.load('./data/test_class_answers.npy')

predictions = model.predict(test_set)
metrics.all_class_metrics(predictions, test_answers)

model.save('./models/class.h5')

t4 = time.time()
# !!! 2.2 Regression model

print('Training regression model')

train_set = np.load('./data/train_regr_set.npy')
train_answers = np.load('./data/train_regr_answers.npy')

model = models.regression_model(train_set[0].shape)
model.fit(train_set, train_answers, batch_size=64, epochs=5, verbose=2, validation_split=0.1)

test_set = np.load('./data/test_regr_set.npy')
test_answers = np.load('./data/test_regr_answers.npy')

predictions = model.predict(test_set)

print('MSE:', metrics.mse(predictions, test_answers))
print('MAE/RMSE:', metrics.rmse(predictions, test_answers))

model.save('./models/regr.h5')

t5 = time.time()

# Time for work
print('Time for extract sequences from input data:', t1-t0)
print('Time for classification dataset', t2-t1)
print('Time for regression dataset', t3-t2)
print('Time for classification model', t4-t3)
print('Time for regression model', t5-t4)
