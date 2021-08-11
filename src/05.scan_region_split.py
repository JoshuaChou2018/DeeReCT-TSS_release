cells='NKtcell colon gastrointestinal leukemia lung myeloma neuroblastoma renal tcell testicular'.split(' ')
step_size=5
for cell in cells:
    file_count=0
    base_count=0
    w=open('scan_regions/{}.RNAseq.bedgraph.merged.{}.bed'.format(cell,file_count),'w')
    for _,line in enumerate(open('scan_regions/{}.RNAseq.bedgraph.merged.bed'.format(cell),'r')):
        if base_count/5>20000000:
            w.close()
            file_count+=1
            base_count=0
            w=open('scan_regions/{}.RNAseq.bedgraph.merged.{}.bed'.format(cell,file_count),'w')
        else:
            base_count+=int(line.split('\t')[2])-int(line.split('\t')[1])
            w.write(line)
    w.close()
