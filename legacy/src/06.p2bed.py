import os
import pickle
import sys

scan_out_path=sys.argv[1] # /home/zhouj0d/c2108/PID1.TSS/DeepTSS/data/human.cell_line.ibex/colon/rexp0/seqonly/rep20/scanall_genebed
scan_save_path=sys.argv[2] #'/home/zhouj0d/c2108/PID1.TSS/DeepTSS/data/human.cell_line.ibex/colon/rexp0/seqonly/rep20/scanall_genebed.bedgraph'
scan_bed_path=sys.argv[3] # '/home/zhouj0d/c2108/PID1.TSS/DeepTSS/data/human.cell_line.ibex/scanall_genebed/gencode.v36lift37.gene.protein_coding.extend.bed'
try:
    step_size=int(sys.argv[4])
except:
    step_size=1
    
print('[INFO] step_size: {}'.format(step_size))

cutoff=0.1
with open(scan_save_path,'w') as w:
    for _,line in enumerate(open(scan_bed_path,'r')):
        #if _==2:
        #    break
        line=line.rstrip('\n').split('\t')
        _chr=line[0]
        _start=int(line[1])
        _end=int(line[2])
        _info=line[3]
        _str=line[-1]
        file_path='{}/{}:{}_{}.p'.format(scan_out_path,_chr,_start,_end)
        if os.path.isfile(file_path):
            print(_,line)
            data=pickle.load(open(file_path,'rb'))
            data=[x[0] for x in data]
            for i in range(len(data)):
                if data[i]>=cutoff:
                    w.write('{}\t{}\t{}\t{:.4f}\t{}\t{}\n'.format(_chr,_start+i*step_size,_start+i*step_size,data[i],_str,_info))
    
