"""
Functions for making dataset for train and test model
"""


import numpy as np
import features as ft
from random import randint


def tss_read(path, sep=','):
    """
    
    :param path: path to file with format Chromosome, Strand, Locus_ID, TSS position
    :param sep: symbol for separate
    :return: sorted by chromosome and tss position array
    """

    tss_array = []
    with open(path) as inf:
        for line in inf:
            point = line.rstrip().split(sep)
            try:
                tss_array += [[point[0], point[1], point[2], int(point[3])]]
            except ValueError:
                continue

    tss_array = sorted(tss_array, key=lambda x: (int(x[0][3:]), x[3]))
    return tss_array


def to_strings(numpy_matrix, path):
    """
    
    :param numpy_matrix: matrix to convert
    :param path: path to new_file
    :param letters_order: 'ACGT' or 'ATCG'
    :return: fasta file with strings
    """
    
    n, l, c = numpy_matrix.shape
    charar = np.chararray((n, l), unicode=True)
    charar[:] = 'N'
    
    charar[np.where(numpy_matrix[:, :, 0] == 1)] = 'A'
    charar[np.where(numpy_matrix[:, :, 1] == 1)] = 'C'
    charar[np.where(numpy_matrix[:, :, 2] == 1)] = 'G'
    charar[np.where(numpy_matrix[:, :, 3] == 1)] = 'T'
    
    with open(path+'.fa', 'w') as ouf:
        for i in range(n):
            ouf.write('>' + str(i+1) + '\n')
            ouf.write(''.join(charar[i])+'\n')
    

def seqs_from_chr(tss_array, seqs_with_tss, gene, chromosome, chromosome_name, nucleotides):
    tss_examples = len(tss_array)
    if gene == tss_examples:
        return gene
    while chromosome_name == tss_array[gene][0]:
        position = tss_array[gene][3]
        seqs_with_tss += [chromosome[position - nucleotides:position + nucleotides]]
        gene += 1
        if gene == tss_examples:
            return gene
    return gene


def sequences(tss_array, dna_seq, split, new_path='./data/', nucleotides=1000):
    """
    
    :param tss_array: tss_array with format [Chromosome, Strand, Locus_ID, TSS position] 
    :param dna_seq: path to fasta file of dna_seq, must be sorted
    :param split: massive with examples arrays for train and test (split_data)
    :param new_path: path to new files
    :param nucleotides: how many nucleotides on the left and right side of TSS will be taken
    :return: new file TSS_seq.fasta for all Locus_ID and new file TSS_pos with positions of TSS for all Locus_ID
    """

    seqs_with_tss = []
    chromosome = ''
    gene = 0
    first_line = True
    max_gene = len(tss_array)

    with open(dna_seq) as inf:
        for line in inf:
            if gene == max_gene:
                break
            elif line[0] == '>':
                if first_line:
                    chromosome_name = line.rstrip()[1:]
                    first_line = False
                else:
                    gene = seqs_from_chr(tss_array, seqs_with_tss, gene, chromosome, chromosome_name, nucleotides)
                    chromosome_name = line.rstrip()[1:]
                    chromosome = ''
            else:
                if chromosome_name != tss_array[gene][0]:
                    continue
                else:
                    chromosome += line.rstrip()
        seqs_from_chr(tss_array, seqs_with_tss, gene, chromosome, chromosome_name, nucleotides)

    print('Writing train_sequences.fa...')
    with open(new_path + 'train_sequences.fa', 'w') as ouf:
        for i in split[0]:
            ouf.write('>' + tss_array[i][2] + '\n')
            if tss_array[i][1] == '+':
                ouf.write(seqs_with_tss[i] + '\n')
            else:
                ouf.write(ft.reverse_compliment(seqs_with_tss[i]) + '\n')

    print('Writing test_sequences.fa...')
    with open(new_path + 'test_sequences.fa', 'w') as ouf:
        for i in split[1]:
            ouf.write('>' + tss_array[i][2] + '\n')
            if tss_array[i][1] == '+':
                ouf.write(seqs_with_tss[i] + '\n')
            else:
                ouf.write(ft.reverse_compliment(seqs_with_tss[i]) + '\n')

    array_length = nucleotides * 2
    print('Writing train_tss_pos.fa...')
    with open(new_path + 'train_tss_pos.fa', 'w') as ouf:
        for i in split[0]:
            ouf.write('>' + tss_array[i][2] + '\n')
            write_array = [0 for i in range(array_length)]
            write_array[nucleotides] = 1

            gene = i
            while True:
                if gene == 0:
                    break
                gene -= 1
                if tss_array[gene][0] == tss_array[i][0]:
                    dif = tss_array[i][3] - tss_array[gene][3]
                    if dif < nucleotides:
                        write_array[nucleotides - dif] = 1
                    else:
                        break
                else:
                    break

            gene = i
            while True:
                if gene == split[0][-1]:
                    break
                gene += 1
                if tss_array[gene][0] == tss_array[i][0]:
                    dif = tss_array[gene][3] - tss_array[i][3]
                    if dif < nucleotides:
                        write_array[nucleotides + dif] = 1
                    else:
                        break
                else:
                    break

            ouf.write(' '.join(map(str, write_array)) + '\n')

    print('Writing test_tss_pos.fa...')
    with open(new_path + 'test_tss_pos.fa', 'w') as ouf:
        for i in split[1]:
            ouf.write('>' + tss_array[i][2] + '\n')
            write_array = [0 for i in range(array_length)]
            write_array[nucleotides] = 1

            gene = i
            while True:
                if gene == 0:
                    break
                gene -= 1
                if tss_array[gene][0] == tss_array[i][0]:
                    dif = tss_array[i][3] - tss_array[gene][3]
                    if dif < nucleotides:
                        write_array[nucleotides - dif] = 1
                    else:
                        break
                else:
                    break

            gene = i
            while True:
                if gene == split[1][-1]:
                    break
                gene += 1
                if tss_array[gene][0] == tss_array[i][0]:
                    dif = tss_array[gene][3] - tss_array[i][3]
                    if dif < nucleotides:
                        write_array[nucleotides + dif] = 1
                    else:
                        break
                else:
                    break

            ouf.write(' '.join(map(str, write_array)) + '\n')


def split_data(tss_array, split=0.1):
    """
    
    :param tss_array: tss_array with format [Chromosome, Strand, Locus_ID, TSS position]
    :param split: part of data for test, default = 0.1
    :return: massive with two arrays: first - data for train, second - for test
    """
    import random

    splitted = [[], []]  # train, test
    chromosome_array = [tss_array[0][0]]
    for i in range(len(tss_array)):
        if tss_array[i][0] != chromosome_array[-1]:
            chromosome_array += [tss_array[i][0]]
        else:
            continue
    print('Chromosomes found:', chromosome_array)
    chromosomes = len(chromosome_array)

    test = [chromosome_array[random.randint(0, chromosomes - 1)]]
    while int(chromosomes * split) < len(test):
        new_test_chr = chromosome_array[random.randint(0, chromosomes - 1)]
        while new_test_chr in test:
            new_test_chr = chromosome_array[random.randint(0, chromosomes - 1)]
        test += [new_test_chr]
    print('Chromosomes for test:', test)

    for i in range(len(tss_array)):
        if tss_array[i][0] in test:
            splitted[1] += [i]
        else:
            splitted[0] += [i]

    return splitted


def regr_assemble(sequences, tss_pos, examples, length, min_pos, max_pos, features=[], skews=[], tata=False):
    """
    
    :param sequences: path to file with seq data (sequences.fasta)
    :param tss_pos: path to file with tss positions data (tss_pos.fasta)
    :param examples: how many examples will be created?
    :param length: length of examples in dataset, must be less then length of sequences
    :param min_pos: minimal position of TSS in example (for example 50 with 512 length)
    :param max_pos: maximal position of TSS in example (for example 462 with 512 length)
    :param features: nucleotide features in dataset? (for example CG, CA, ATG and other)
    :param skews: skew in dataset? (for example CG)
    :param tata: TATAbox feature in dataset?
    :return: return numpy dataset and answers for work with regression models
    """
    import random
    import numpy as np

    dataset = np.zeros((examples, 4 + len(features) + len(skews) + tata, length))
    answers = np.zeros((examples))

    print('Regression dataset assembler works...', end=' ')
    print('Features: A, C, G, T,', ' '.join(features), end=' ')
    for i in skews:
        print(i + 'skew', end=' ')
    if tata:
        print('TATAbox')

    seq_data = open(sequences).readlines()
    tss_data = open(tss_pos).readlines()
    all_tss = len(seq_data) / 2
    middle = len(seq_data[1].rstrip()) // 2
    min_start_pos = middle - max_pos
    max_start_pos = middle - min_pos

    for example in range(examples):
        gene = random.randint(0, all_tss - 1) * 2 + 1
        while sum(map(int, tss_data[gene].rstrip().split())) > 1:
            gene = random.randint(0, all_tss - 1) * 2 + 1

        start_pos = random.randint(min_start_pos, max_start_pos)
        end_pos = start_pos + length
        answers[example] = middle - start_pos

        dataset[example] = features_work(seq_data[gene][start_pos:end_pos], features=features, skews=skews, tata=tata)

    return shuffle_arrays(np.rollaxis(dataset, 1, 3), answers)


def class_assemble(sequences, beyond_genes_seqs, tss_pos, examples, length, min_pos, max_pos, features=[], skews=[],
                   tata=False):
    """
    
    :param sequences: path to file with seq data (sequences.fa)
    :param beyond_genes: path to file with seqs of beyond genes examples (sequences.fa)
    :param tss_pos: path to file with tss positions data (tss_pos.fa)
    :param examples: how many examples will be created? must be //2 = 0
    :param length: length of examples in dataset, must be less then length of sequences
    :param min_pos: min position of TSS in example (for example 50 with 512 length)
    :param max_pos: max position of TSS in example (for example 462 with 512 length)
    :param features: nucleotide features in dataset? (for example CG, CA, ATG and other)
    :param skews: skew in dataset? (for example CG)
    :param tata: TATAbox feature in dataset?
    :return: return numpy dataset and answers for work with classification models, 4th part of examples will be from 
    beyond genes space
    """
    dataset = np.zeros((examples, 4 + len(features) + len(skews) + tata, length))
    answers = np.array([i for i in [1, 0] for j in range(examples // 2)])
    positions = np.zeros(examples)

    print('Classification dataset assembler works...', end=' ')
    print('Features: A, C, G, T,', ' '.join(features), end=' ')
    for i in skews:
        print(i + 'skew', end=' ')
    if tata:
        print('TATAbox')

    seq_data = open(sequences).readlines()
    beyond_genes_data = open(beyond_genes_seqs).readlines()
    tss_data = open(tss_pos).readlines()
    all_tss = len(seq_data) / 2
    all_beyond = len(beyond_genes_data) / 2
    input_length = len(seq_data[1].rstrip())
    middle = input_length // 2
    min_start_pos = middle - max_pos
    max_start_pos = middle - min_pos

    for example in range(examples // 2):
        gene = randint(0, all_tss - 1) * 2 + 1
        start_pos = randint(min_start_pos, max_start_pos)
        end_pos = start_pos + length
        positions[example] = middle - start_pos
        while sum(map(int, tss_data[gene].rstrip().split())) > 1:
            gene = randint(0, all_tss - 1) * 2 + 1

        dataset[example] = features_work(seq_data[gene][start_pos:end_pos], features=features, skews=skews, tata=tata)

    for example in range(examples // 2, examples // 4 * 3):
        gene = randint(0, all_beyond - 1) * 2 + 1
        dataset[example] = features_work(beyond_genes_data[gene][:length], features=features, skews=skews, tata=tata)

    for example in range(examples // 4 * 3, examples):
        gene = randint(0, all_tss - 1) * 2 + 1
        while sum(map(int, tss_data[gene].rstrip().split())) > 1:
            gene = randint(0, all_tss - 1) * 2 + 1

        if randint(0, 1) == 0:
            start_pos = randint(0, middle - min_pos - length)
            end_pos = start_pos + length
            dataset[example] = features_work(seq_data[gene][start_pos:end_pos], features=features,
                                             skews=skews, tata=tata)
        else:
            start_pos = randint(middle+min_pos, input_length-length)
            end_pos = start_pos + length
            dataset[example] = features_work(seq_data[gene][start_pos:end_pos], features=features,
                                             skews=skews, tata=tata)
    
    s = np.arange(answers.shape[0])
    np.random.shuffle(s)

    return np.rollaxis(dataset, 1, 3)[s], answers[s]


def beyond_genes(tss_array, dna_seq, split, examples_train, examples_test, new_path='./data/', nucleotides=512,
                 distance=2000):
    """
    
    :param tss_array: tss_array with format [Chromosome, Strand, Locus_ID, TSS position] 
    :param dna_seq: path to fasta file of dna_seq, must be sorted
    :param split: massive with examples arrays for train and test (split_data)
    :param examples_train: examples for train data
    :param examples_test: examples for test data
    :param new_path: path to new files
    :param nucleotides: how many nucleotides on the left and right side of TSS will be taken
    :param distance: min distance to tss
    :return: new files: train_nogenes_seqs.fa, test_nogenes_seqs.fa, half of examples will be reversed
    """
    chromosomes = []
    lines = open(dna_seq).readlines()

    for i in range(len(lines)):
        if lines[i][0] == '>':
            chromosomes += [[lines[i][1:].strip(), i]]
    length_str_in_line = len(lines[1].rstrip())
    length_split_train = len(split[0])
    length_split_test = len(split[1])

    with open(new_path + 'train_nogenes_seqs.fa', 'w') as ouf:
        for example in range(0, examples_train // 2):

            # choose random space between genes
            randgene = split[0][randint(0, length_split_train-2)]
            while tss_array[randgene + 1][3] - tss_array[randgene][3] < distance * 2 + nucleotides:
                randgene = split[0][randint(0, length_split_train-2)]

            randpos = randint(tss_array[randgene][3] + distance, tss_array[randgene+1][3] - distance - nucleotides)

            # write data
            info = '>' + 'Space between ' + tss_array[randgene][2] + ' and ' + tss_array[randgene + 1][2] \
                   + ' strand + ' + 'in position ' + str(randpos) + '\n'
            ouf.write(info)

            for i in chromosomes:
                if tss_array[randgene][0] == i[0]:
                    chr_start = i[1] + 1
                    break

            first_str = chr_start + randpos // length_str_in_line
            last_str = chr_start + (randpos + nucleotides) // length_str_in_line
            if (randpos + nucleotides) % length_str_in_line == 0:
                last_str -= 1

            seq = lines[first_str][randpos % length_str_in_line:].rstrip()
            for i in range(first_str+1, last_str):
                seq += lines[i].rstrip()
            seq += lines[last_str][:nucleotides-len(seq)].rstrip()

            ouf.write(seq + '\n')

        for example in range(examples_train//2, examples_train):

            # choose random space between genes
            randgene = split[0][randint(0, length_split_train-2)]
            while tss_array[randgene + 1][3] - tss_array[randgene][3] <= distance * 2 + nucleotides:
                randgene = split[0][randint(0, length_split_train-2)]

            randpos = randint(tss_array[randgene][3] + distance, tss_array[randgene+1][3] - distance - nucleotides)

            # write data
            info = '>' + 'Space between ' + tss_array[randgene][2] + ' and ' + tss_array[randgene + 1][2] \
                   + ' strand - ' + 'in position ' + str(randpos) + '\n'
            ouf.write(info)

            for i in chromosomes:
                if tss_array[randgene][0] == i[0]:
                    chr_start = i[1] + 1
                    break

            first_str = chr_start + randpos // length_str_in_line
            last_str = chr_start + (randpos + nucleotides) // length_str_in_line
            if (randpos + nucleotides) % length_str_in_line == 0:
                last_str -= 1

            seq = lines[first_str][randpos % length_str_in_line:].rstrip()
            for i in range(first_str + 1, last_str):
                seq += lines[i].rstrip()
            seq += lines[last_str][:nucleotides - len(seq)].rstrip()

            ouf.write(ft.reverse_compliment(seq) + '\n')

    with open(new_path + 'test_nogenes_seqs.fa', 'w') as ouf:
        for example in range(0, examples_test // 2):

            # choose random space between genes
            randgene = split[1][randint(0, length_split_test-2)]
            while tss_array[randgene + 1][3] - tss_array[randgene][3] <= distance * 2 + nucleotides:
                randgene = split[1][randint(0, length_split_test-2)]

            randpos = randint(tss_array[randgene][3] + distance, tss_array[randgene+1][3] - distance - nucleotides)

            # write data
            info = '>' + 'Space between ' + tss_array[randgene][2] + ' and ' + tss_array[randgene + 1][2] \
                   + ' strand + ' + 'in position ' + str(randpos) + '\n'
            ouf.write(info)

            for i in chromosomes:
                if tss_array[randgene][0] == i[0]:
                    chr_start = i[1] + 1
                    break

            first_str = chr_start + randpos // length_str_in_line
            last_str = chr_start + (randpos + nucleotides) // length_str_in_line
            if (randpos + nucleotides) % length_str_in_line == 0:
                last_str -= 1

            seq = lines[first_str][randpos % length_str_in_line:].rstrip()
            for i in range(first_str + 1, last_str):
                seq += lines[i].rstrip()
            seq += lines[last_str][:nucleotides - len(seq)].rstrip()

            ouf.write(seq+'\n')

        for example in range(examples_test // 2, examples_test):

            # choose random space between genes
            randgene = split[1][randint(0, length_split_test-2)]
            while tss_array[randgene + 1][3] - tss_array[randgene][3] <= distance * 2 + nucleotides:
                randgene = split[1][randint(0, length_split_test-2)]

            randpos = randint(tss_array[randgene][3] + distance, tss_array[randgene+1][3] - distance - nucleotides)

            # write data
            info = '>' + 'Space between ' + tss_array[randgene][2] + ' and ' + tss_array[randgene + 1][2] \
                   + ' strand - ' + 'in position ' + str(randpos) + '\n'
            ouf.write(info)

            for i in chromosomes:
                if tss_array[randgene][0] == i[0]:
                    chr_start = i[1] + 1
                    break

            first_str = chr_start + randpos // length_str_in_line
            last_str = chr_start + (randpos + nucleotides) // length_str_in_line
            if (randpos + nucleotides) % length_str_in_line == 0:
                last_str -= 1

            seq = lines[first_str][randpos % length_str_in_line:].rstrip()
            for i in range(first_str + 1, last_str):
                seq += lines[i].rstrip()
            seq += lines[last_str][:nucleotides - len(seq)].rstrip()

            ouf.write(ft.reverse_compliment(seq)+'\n')

    print('Beyond genes space data is ready')


def features_work(seq, features=[], skews=[], tata=False):
    example_matrix = np.zeros((4 + len(features) + len(skews) + tata, len(seq)))

    example_matrix[:4] = ft.nuc_to_arr(seq)
    for feature in range(len(features)):
        example_matrix[4 + feature] = ft.nucleotide_feature_array(example_matrix[:4], features[feature])

    for skew in range(len(skews)):
        example_matrix[4 + len(features) + skew] = ft.skew_array(example_matrix[:4], skews[skew][0], skews[skew][1])

    if tata:
        example_matrix[-1] = ft.tata_box_array(example_matrix[:4])

    return example_matrix


def shuffle_arrays(array_1, array_2):

    s = np.arange(array_1.shape[0])
    np.random.shuffle(s)

    return array_1[s], array_2[s]
