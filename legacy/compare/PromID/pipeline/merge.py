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
import time

def shorten(inp, s):
    inp = re.sub(r'\s+', '', inp)
    if(s):
        inp = inp[int(sys.argv[3]):int(sys.argv[3])+int(sys.argv[4])]
    inp = inp+"\n"
    return inp

def read(inp, s):
    data = []
    seq = ""
    with open(inp) as f:
        for line in f:
            if(line.startswith(">")):
                if(len(seq)!=0):
                    data.append(shorten(seq, s))
                    seq=""                    
                continue                
            else:
                seq+=line
    if(len(seq)!=0):
        data.append(shorten(seq, s))
        seq="" 
    return data

def write(file, data):
    fh = open(file, "w")
    for i in range(len(data)):
        fh.write("> ")
        fh.write("\n")
        fh.write(data[i])
    fh.close()

#np.random.seed(2504) 

total = len(sys.argv)
#if total<4:
#    print('USAGE: <input file 1> <input file 2> <value between 0 and 1>')
#    exit(0)

data1 = read(str(sys.argv[1]), False)
data2 = read(str(sys.argv[2]), False)

write(str(time.time()), data2)

data1 = np.array(data1)
data2 = np.array(data2)

size1 = len(data1) + len(data2)
fdata = list(set().union(data1, data2))
size2 = len(fdata) 
print("Duplicates: " + str(size1 - size2))
np.random.shuffle(fdata)
#end = len(data1) + int(sys.argv[4]) [:len(data1)]
#[:min(len(fdata), int(sys.argv[4]))]
write(sys.argv[3], fdata)
