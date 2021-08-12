cat << EOF >log/p2bed_$1.sh
#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J log/p2bed_$1
#SBATCH -o log/p2bed_$1.out
#SBATCH -e log/p2bed_$1.err
#SBATCH --time=10:00:00
#SBATCH --mem=150G

python /home/zhouj0d/c2066/DeeReCT-TSS_release/src/06.p2bed.py \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/scan_regions_out \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/scan_regions.bedgraph \
/home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/scan_regions/$1.RNAseq.bedgraph.merged.bed

EOF

sbatch log/p2bed_$1.sh
