scanbed=/home/zhouj0d/c2066/DeeReCT-TSS_release/data/TCGA-AA-3517-11A-01R-A32Z-07/scan_regions/TCGA-AA-3517-11A-01R-A32Z-07.RNAseq.bedgraph.merged.0.bed
base=$(basename $scanbed)

cat << EOF >log/scanapp_$base.sh
#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J log/scanapp_$base
#SBATCH -o log/scanapp_$base.out
#SBATCH -e log/scanapp_$base.err
#SBATCH --time=12:00:00
#SBATCH --mem=100G

#run the application:
date
cd /home/zhouj0d/c2066/DeeReCT-TSS_release/src
conda activate tss
python 05.scan.app.v2.py $scanbed \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/TCGA-AA-3517-11A-01R-A32Z-07 \
/home/zhouj0d/c2066/DeeReCT-TSS_release/model/colon_model/model_best.npz \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/TCGA-AA-3517-11A-01R-A32Z-07/rnaseq/Aligned.sortedByCoord.out.bam.+.depth \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/TCGA-AA-3517-11A-01R-A32Z-07/rnaseq/Aligned.sortedByCoord.out.bam.-.depth \
/home/zhouj0d/c2066/DeeReCT-TSS_release/ref/hg38/hg38.fa \
5 0 0.1
date
EOF

sbatch log/scanapp_$base.sh

