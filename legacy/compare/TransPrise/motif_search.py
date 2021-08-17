import models
import numpy as np
import os
import weblogolib
import features as ft
from scipy import ndimage
import matplotlib.pyplot as plt

def weights_modification(weights):
    """
    
    :param weights: weights matrix 
    :return: Position-Weight Matrix
    """
    weights[np.where(weights < 0)] = 0
    weights[np.where(np.sum(weights, axis=1)==0)] = [0.25, 0.25, 0.25, 0.25]
    return weights/np.sum(weights, axis=1, keepdims=True)


def seqlogo(source_name, list_weights):
    """

    :param source_name: string source name for folder name
    :param list_weights: list of weights from different layers
    :return: sequence logos in folders for all filters
    """

    for l_ in range(len(list_weights)):
        layer_w = np.moveaxis(list_weights[l_], 2, 0)
        length = layer_w.shape[1]
        path = './' + source_name + '_' + str(length) + '/'
        os.makedirs(path[:-1], exist_ok=True)

        for w_ in range(layer_w.shape[0]):
            data = weblogolib.LogoData.from_counts('ACGT', weights_modification(layer_w[w_]))
            options = weblogolib.LogoOptions(fineprint = False, logo_title='', color_scheme = weblogolib.classic,
                                             stack_width = weblogolib.std_sizes["large"], logo_start=1, logo_end=length, resolution=600)
            logo_format = weblogolib.LogoFormat(data, options)
            f=open(path+str(w_+1)+'.png', 'wb')
            f.write(weblogolib.png_formatter(data, logo_format))
            f.close()


def convolution(to_convolve, w_):
    W_ = weights_modification(w_.T).T
    result = np.zeros(to_convolve.shape[2])
    w_shape = W_.shape[1]
    max_value = np.sum(W_[np.argmax(W_, axis=0)])
    examples = to_convolve.shape[0]

    for step in range(to_convolve.shape[2]-w_shape):
        result[step] = np.sum(np.multiply(to_convolve[:, :, step:step+w_shape], W_))/max_value
    return result/examples


def more_then_one(seqs, source_name, list_weights, marker, f):
    """

    :param seqs: path to seqs
    :param source_name: string source name for folder name
    :param list_weights: list of weights from different layers
    :return: plots where conv > 1
    """
    seq_data = open(seqs).readlines()
    examples = len(seq_data) // 2
    length = len(seq_data[1].rstrip())
    dataset = np.zeros((examples, 4, length))
    f = open('./'+f+'.fa', 'w')

    for ex in range(examples):
        seq = seq_data[ex*2+1].rstrip()
        dataset[ex] = ft.nuc_to_arr(seq)

    for l_ in range(len(list_weights)):
        layer_w = np.moveaxis(list_weights[l_], 2, 0)
        length = layer_w.shape[1]
        #path = './'+source_name+'_'+str(length)+'/'

        for w_ in range(layer_w.shape[0]):
            answer = convolution(dataset, np.moveaxis(layer_w[w_], 1, 0))
            answer[-length:] = answer[-length-1]
            f.write('>'+str(length)+'_'+str(w_)+'\n')
            f.write(' '.join(map(str, answer)))
            f.write('\n')
            #plt.figure(figsize=(20,10))
            #plt.plot(np.arange(-1000, 1000), answer, 'b')
            #plt.ylabel('Frequency')
            #plt.xlabel('Position')
            #plt.savefig(path+str(w_+1)+'_'+marker, dpi=500)
            #plt.close()
    
    f.close()

    


model_regression = models.regression_model((512, 4))
model_classification = models.classification_model((512, 4))

model_regression.load_weights('./models/regression.h5')
model_classification.load_weights('./models/classification.h5')

#seqlogo('regression', model_regression.get_weights()[:4])
#seqlogo('classification', model_classification.get_weights()[:4])

more_then_one('./data/train_sequences.fa', 'regression', model_regression.get_weights()[:4], 'freq', 'regression')
more_then_one('./data/train_sequences.fa', 'classification', model_classification.get_weights()[:4], 'freq', 'classification')
