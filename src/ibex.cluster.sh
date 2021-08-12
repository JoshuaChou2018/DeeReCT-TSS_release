cat << EOF >log/cluster_$1.sh
#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J log/cluster_$1
#SBATCH -o log/cluster_$1.out
#SBATCH -e log/cluster_$1.err
#SBATCH --time=10:00:00
#SBATCH --mem=150G

date
module load bedtools
bash /home/zhouj0d/c2066/DeeReCT-TSS_release/src/06.clusterTSS.sh \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/scan_regions.bedgraph \
$1
mv /home/zhouj0d/c2066/DeeReCT-TSS_release/src/$1.predicted.TSS.withpval /home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/$1.predicted.TSS.withpval
date

EOF

sbatch log/cluster_$1.sh
