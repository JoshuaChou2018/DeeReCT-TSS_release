import models
import numpy as np


def dataset(path):
    fasta = open(path).readlines()
    locs_names, seqs_set, first_row = [], [], True

    for row in fasta:
        if row[0] == '>':
            locs_names.append([row.rstrip(), len(seqs_set)])
            if first_row:
                first_row = False
            else:
                locs_names[-2].append(len(seqs_set))
        else:
            seqs_set.append(row.rstrip().lower())
    locs_names[-1].append(len(seqs_set))

    del fasta
    return locs_names, seqs_set


def to_matrix(seq):
    charar = np.chararray(len(seq), unicode=True)
    charar[:] = list(seq)
    output_ar = np.zeros((len(seq), 4))
    
    output_ar[np.where(charar == 'a')] = [1, 0, 0, 0]
    output_ar[np.where(charar == 'c')] = [0, 1, 0, 0]
    output_ar[np.where(charar == 'g')] = [0, 0, 1, 0]
    output_ar[np.where(charar == 't')] = [0, 0, 0, 1]
    
    return output_ar


def batch(matrix):
    count = (matrix.shape[0]-512)//4+1
    indexes, _ = np.meshgrid(np.arange(512), np.arange(count))
    indexes = indexes + np.expand_dims(np.linspace(0, 4*(count-1), count), axis=1).astype(np.int32)
    return matrix[indexes]


def analysis(seq, pos, thres):
    if len(seq) < 650:
        return 0
    btch = batch(to_matrix(seq))
    clss = model_c.predict(btch)
    idxs = np.arange(btch.shape[0])[np.where(clss>=0.5)[0]]
    regression = np.zeros(btch.shape[0])
    ideal = np.linspace(400, 300, 26)
    
    results = []
    if len(idxs) == 0:
        return 0
    else:
        regression[idxs] = np.floor(np.squeeze(model_r.predict(btch[idxs])))
        if len(regression) < 26:
            return 0
        else:
            for idx in range(regression.shape[0]-26):
                if regression[idx:idx+26].sum() == 0:
                    continue
                else:
                    l1norm = np.linalg.norm(ideal-regression[idx:idx+26], ord=1)
                    if l1norm <= thres:
                        results.append([idx*4+pos+400, l1norm])
    return results


path = input('Path to file: ')
thres = int(input('Input threshold for L1Norm: ')) # default 100
output = open('./output.txt', 'w')
output.write('loc pos similarity \n')

# models_load
shape = (512, 4)
model_c = models.classification_model(shape)
model_r = models.regression_model(shape)
model_c.load_weights('./models/classification.h5')
model_r.load_weights('./models/regression.h5')

# dataset_make
locs_names, seqs_set = dataset(path)

# analysis
for loc in locs_names:
    # read loc
    start, end, seq, seq_len, pos, results = loc[1], loc[2], '', 0, 0, []
    for row in range(start, end):
        if row%10000 == 0:
            print('loc', loc[0], 'position', pos)
        seq += seqs_set[row]
        seq_len += len(seqs_set[row])
        if seq_len >= 2000:
            result = analysis(seq[:2000], pos, thres)
            if result != 0:
                results += result
            seq = seq[1750:]
            seq_len -= 1750
            pos += 1750
        else:
            continue
    # loc analysis
    result = analysis(seq, pos, thres)
    if result != 0:
        results += result

    # result interpretation
    max_len = len(results) - 1
    current_pos = 0
    current_score = thres
    for ind, val in enumerate(results):
        current_delta = abs(val[0] - current_pos)
        if current_delta > 100 and current_score < thres or ind == max_len:
            output.write('%s %s %s \n' % (loc[0], current_pos, current_score))
            current_pos, current_score = val[0], val[1]
        elif current_delta > 100:
            current_pos, current_score = val[0], val[1]
        elif current_delta <= 100 and val[1] < current_score:
            current_pos, current_score = val[0], val[1]
        else:
            continue

output.close()
