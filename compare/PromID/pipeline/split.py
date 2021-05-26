#!/usr/bin/env python
import numpy as np
from math import sqrt
import numpy as np
from numpy import zeros
import sys
import re
import math
import os
import pdb
import string

def shorten(inp, s):
    if(s):
        inp = re.sub(r'\s+', '', inp)
        inp = inp[int(sys.argv[3]):int(sys.argv[3])+int(sys.argv[4])]
        inp = inp+"\n"
    return inp

def read(inp, s):
    data = []
    names = []
    seq = ""
    with open(inp) as f:
        for line in f:
            if(line.startswith(">")):
                names.append(line.strip())
                if(len(seq)!=0):
                    data.append(shorten(seq, s))
                    seq=""                    
                continue                
            else:
                seq+=line
    if(len(seq)!=0):
        data.append(shorten(seq, s))
        seq="" 
    return data, names

def write(file, names, data):
    fh = open(file, "w")
    for i in range(len(data)):
        fh.write(names[i])
        fh.write("\n")
        fh.write(data[i])
    fh.close()

#np.random.seed(2504) 

total = len(sys.argv)
#if total<4:
#    print('USAGE: <input file 1> <input file 2> <value between 0 and 1>')
#    exit(0)

data1, names1 = read(str(sys.argv[1]), True)
data2, names2 = read(str(sys.argv[1]), False)
data1 = np.array(data1)
names1 = np.array(names1)
data2 = np.array(data2)
names2 = np.array(names2)
inds = names1.argsort()
names1 = names1[inds]
data1 = data1[inds]

inds = names2.argsort()
names2 = names2[inds]
data2 = data2[inds]

rng_state = np.random.get_state()
np.random.shuffle(data1)
np.random.set_state(rng_state)
np.random.shuffle(names1)
np.random.set_state(rng_state)
np.random.shuffle(data2)
np.random.set_state(rng_state)
np.random.shuffle(names2)
size = len(data1)
s1 = int(float(sys.argv[2]) * size)
s2 = size - s1
print("File 1 - " + str(s1) + " sequences")
print("File 2 - " + str(s2) + " sequences")
write("p_1", names1[:s1], data1[:s1])
write("p_2", names1[s1:], data1[s1:])
write("lp_1", names2[:s1], data2[:s1])
write("lp_2", names2[s1:], data2[s1:])
