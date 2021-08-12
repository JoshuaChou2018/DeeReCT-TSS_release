cat << EOF >log/getscanregion_$1.sh
#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J log/getscanregion_$1
#SBATCH -o log/getscanregion_$1.out
#SBATCH -e log/getscanregion_$1.err
#SBATCH --time=2:00:00
#SBATCH --mem=20G

#run the application:

date

mkdir -p /home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/scan_regions
bash /home/zhouj0d/c2066/DeeReCT-TSS_release/src/05.getScanRegion.sh /home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/rnaseq/Aligned.sortedByCoord.out.bam /home/zhouj0d/c2066/DeeReCT-TSS_release/data/$1/scan_regions/$1 1 /home/zhouj0d/c2066/DeeReCT-TSS_release/ref/gencode.v38.pcg.extups5k.bed

date

EOF

sbatch log/getscanregion_$1.sh
