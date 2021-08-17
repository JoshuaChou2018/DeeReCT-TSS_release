cells='TCGA-AA-3517-11A-01R-A32Z-07'.split(' ')
step_size=5
for cell in cells:
    file_count=0
    base_count=0
    w=open('../data/{}/scan_regions/{}.RNAseq.bedgraph.merged.{}.bed'.format(cell,cell,file_count),'w')
    for _,line in enumerate(open('../data/{}/scan_regions/{}.RNAseq.bedgraph.merged.bed'.format(cell,cell),'r')):
        if base_count/step_size>20000000/5:
            w.close()
            file_count+=1
            base_count=0
            w=open('../data/{}/scan_regions/{}.RNAseq.bedgraph.merged.{}.bed'.format(cell,cell,file_count),'w')
        else:
            base_count+=int(line.split('\t')[2])-int(line.split('\t')[1])
            w.write(line)
    w.close()
