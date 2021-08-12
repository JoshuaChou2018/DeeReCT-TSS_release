for i in {0..0}
do

cat << EOF >log/scanapp_$1_${i}.sh
#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J log/scanapp_$1_${i}
#SBATCH -o log/scanapp_$1_${i}.out
#SBATCH -e log/scanapp_$1_${i}.err
#SBATCH --time=12:00:00
#SBATCH --mem=100G

#run the application:
date
cd /home/zhouj0d/c2066/DeeReCT-TSS_release/src
conda activate tss
python 05.scan.app.py $i \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1 \
/home/zhouj0d/c2066/DeeReCT-TSS_release/model/colon_model/model_best.npz \
/scan_regions_out \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/scan_regions/$1.RNAseq.bedgraph.merged \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/rnaseq/Aligned.sortedByCoord.out.bam.+.depth \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/rnaseq/Aligned.sortedByCoord.out.bam.-.depth \
/home/zhouj0d/c2066/DeeReCT-TSS_release/ref/hg38/hg38.fa \
5 0
date
EOF

sbatch log/scanapp_$1_${i}.sh

done
