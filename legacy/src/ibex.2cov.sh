cat << EOF >log/2cov_$1_rnaseq.sh
#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J log/2cov_$1_rnaseq
#SBATCH -o log/2cov_$1_rnaseq.out
#SBATCH -e log/2cov_$1_rnaseq.err
#SBATCH --time=12:00:00
#SBATCH --mem=200G

#run the application:

date
cd /home/zhouj0d/c2066/TCGA/analysis/COAD/bam
mkdir -p /home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/rnaseq
ln -s /home/zhouj0d/c2066/TCGA/analysis/COAD/bam/$1.Aligned.sortedByCoord.out.bam /home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/rnaseq/Aligned.sortedByCoord.out.bam
bash /home/zhouj0d/c2066/DeeReCT-TSS_release/src/01.rnaseq2cov.sh /home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/rnaseq/Aligned.sortedByCoord.out.bam
date

EOF

sbatch log/2cov_$1_rnaseq.sh



