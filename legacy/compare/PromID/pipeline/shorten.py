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
import random

def shorten(inp):
    inp = re.sub(r'\s+', '', inp)
    st = 0
    while True: 
        st = random.randint(0, len(inp) - int(sys.argv[3]))
        if(st<int(sys.argv[2]) - int(sys.argv[4]) or st > int(sys.argv[2]) + int(sys.argv[4])):
            break
    inp = inp[st:st+int(sys.argv[3])]
    inp = inp+"\n"
    return inp

def read(inp):
    data = []
    names = []
    seq = ""
    with open(inp) as f:
        for line in f:
            if(line.startswith(">")):
                names.append(line.strip())
                if(len(seq)!=0):
                    data.append(shorten(seq))
                    seq=""                    
                continue                
            else:
                seq+=line
    if(len(seq)!=0):
        data.append(shorten(seq))
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
if total<2:
    print('USAGE: <input file 1> <input file 2> <value between 0 and 1>')
    exit(0)

data1, names1 = read(str(sys.argv[1]))

write("n_1", names1, data1)
