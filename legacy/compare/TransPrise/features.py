"""
Module with functions for different features from dna sequence
"""
import numpy as np


def reverse_compliment(seq):
    """
    
    :param seq: dna sequence 5'-3' 
    :return: compliment reversed dna sequence 5'-3'
    """
    seq = seq.upper()
    rev_seq = ''

    for nuc in seq:
        for letter in range(5):
            if nuc == 'ACGTN'[letter]:
                rev_seq += 'TGCAN'[letter]
                break
            else:
                continue

    return rev_seq[::-1]


def nucleotide_feature(seq, nucleotide):
    """
    
    :param seq: dna sequence 5'-3' 
    :param nucleotide: nucleotide feature to return, it can be mono-, di- or more- nucleotide
    :return: matrix of size len(seq) with 1 in positions with nucleotide and 0 else.
    """
    seq = seq.upper()
    nucleotide = nucleotide.upper()
    seq_size = len(seq)
    nuc_size = len(nucleotide)
    matrix = np.zeros(seq_size)

    for pos in range(0, seq_size-nuc_size+1):
        if seq[pos:pos+nuc_size] == nucleotide:
            matrix[pos] = 1
        else:
            continue

    return matrix


def tata_box(seq):
    """
    
    :param seq: dna sequence 5'-3'
    :return: matrix of size len(seq) with 1 in positions of start TATA-box and 0 else.
    """
    seq = seq.upper()
    seq_size = len(seq)
    matrix = np.zeros(seq_size)

    for pos in range(seq_size-6):
        if seq[pos:pos+4] == 'TATA':
            if seq[pos+4:pos+7] in ['AAA', 'AAT', 'TAA', 'TAT']:
                matrix[pos] = 1
            else:
                continue
        else:
            continue

    return matrix


def skew(seq, nuc_1, nuc_2):
    """
    
    :param seq: dna sequence 5'-3'
    :param nuc_1: First nuc for skew
    :param nuc_2: Second nuc for skew
    :return: if nuc_1 is C and nuc_2 is G, it would return matrix of CGSkew for given seq
    """
    seq = seq.upper()
    nuc_1 = nuc_1.upper()
    nuc_2 = nuc_2.upper()
    nuc_1_counter = 1
    nuc_2_counter = 2
    matrix = []

    for nuc in seq:
        if nuc == nuc_1:
            nuc_1_counter += 1
        elif nuc == nuc_2:
            nuc_2_counter += 1

        matrix += [(nuc_1_counter-nuc_2_counter)/(nuc_1_counter+nuc_2_counter)]

    return matrix


def reverse_compliment_array(matrix):
    """
    
    :param matrix: numpy array of ACGT. 1st row A, 2nd row C, 3rd row G, 4th row T
    :return: new matrix of reverse compliment
    """
    new_matrix = np.zeros(matrix.shape)

    new_matrix[0] = matrix[3, ::-1]
    new_matrix[1] = matrix[2, ::-1]
    new_matrix[2] = matrix[1, ::-1]
    new_matrix[3] = matrix[0, ::-1]

    return new_matrix


def nucleotide_feature_array(matrix, nucleotide):
    """
    
    :param matrix: numpy array of ACGT. 1st row A, 2nd row C, 3rd row G, 4th row T
    :param nucleotide: nucleotide feature to return, it can be mono-, di- or more- nucleotide
    :return: new matrix of nucleotide feature
    """
    length_1 = len(matrix[0])
    length_2 = len(nucleotide)
    new_matrix = np.zeros(length_1)
    feature = nuc_to_arr(nucleotide)

    for pos in range(length_1-length_2+1):
        if np.array_equal(matrix[:, pos:pos+length_2], feature):
            new_matrix[pos] = 1
        else:
            continue

    return new_matrix


def tata_box_array(matrix):
    """
    
    :param matrix: numpy array of ACGT. 1st row A, 2nd row C, 3rd row G, 4th row T
    :return: new matrix of tata_box feature
    """
    length = len(matrix[0])
    tata_array = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]).reshape((4, 4))
    tata_array_1 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((4, 3))
    tata_array_2 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape((4, 3))
    tata_array_3 = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape((4, 3))
    tata_array_4 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]).reshape((4, 3))
    tata_feat = np.zeros(length)

    for pos in range(length-6):
        if np.array_equal(matrix[:, pos:pos+4], tata_array):
            if np.array_equal(matrix[:, pos+4:pos+7], tata_array_1) or \
                    np.array_equal(matrix[:, pos + 4:pos + 7], tata_array_2) or \
                    np.array_equal(matrix[:, pos + 4:pos + 7], tata_array_3) or \
                    np.array_equal(matrix[:, pos + 4:pos + 7], tata_array_4):
                tata_feat[pos] = 1
            else:
                continue
        else:
            continue

    return tata_feat


def skew_array(matrix, nuc_1, nuc_2):
    """
    
    :param matrix: numpy array of ACGT. 1st row A, 2nd row C, 3rd row G, 4th row T
    :param nuc_1: First nuc for skew, must be a letter
    :param nuc_2: Second nuc for skew, must be a letter
    :return: if nuc_1 is C and nuc_2 is G, it would return matrix of CGSkew for given seq
    """
    length = len(matrix[0])
    skew_matrix = np.zeros(length)
    nuc_1 = nuc_1.upper()
    nuc_2 = nuc_2.upper()
    nuc_1_counter = 1
    nuc_2_counter = 1

    for letter in range(4):
        if nuc_1 == 'ACGT'[letter]:
            nuc_1 = letter
        elif nuc_2 == 'ACGT'[letter]:
            nuc_2 = letter
        else:
            continue

    for pos in range(length):
        nuc_1_counter += matrix[nuc_1, pos]
        nuc_2_counter += matrix[nuc_2, pos]
        skew_matrix[pos] = (nuc_1_counter-nuc_2_counter)/(nuc_1_counter+nuc_2_counter)

    return skew_matrix


def nuc_to_arr(seq):
    """
    
    :param seq: sequence for making matrix 4xlen(seq)
    :return: numpy array with positions of letters in nucleotide
    """
    length = len(seq)
    feature = np.zeros((4, length))
    seq = seq.upper()

    for pos in range(length):
        if seq[pos] == 'A':
            feature[0, pos] = 1
        elif seq[pos] == 'C':
            feature[1, pos] = 1
        elif seq[pos] == 'G':
            feature[2, pos] = 1
        elif seq[pos] == 'T':
            feature[3, pos] = 1
        else:
            continue

    return feature
