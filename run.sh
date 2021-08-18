#!/bin/bash
:<<USE
USE

function usage () {
        echo
        echo
		echo Usage:run.sh [bam_file] [bed_regions] [model] [genome_fa] [output_directory] [0/1/2 ifstranded] [thread]
        echo
		echo
}

if [ -z $6 ]; then
        usage
        exit
fi

bam=${1}
bed_reg=${2}
model=${3}
genome=${4}
out_dir=${5}
IFSTR=${6}
thread=${7}

##
#conda activate tss
#module load bedtools
##

echo processing bam file $bam ...
#:<<FIN
#date
if [ $IFSTR == 1 ]; then
        echo the sequencing data is from sense strand ... 
        genomeCoverageBed -ibam $bam -split -dz -strand + | awk '{OFS="\t"}{print $1,$2,$2+1,$1":"$2,$3,"+"}' | grep "chr" > $out_dir/RNAseq.depth.bed
        genomeCoverageBed -ibam $bam -split -dz -strand - | awk '{OFS="\t"}{print $1,$2,$2+1,$1":"$2,$3,"-"}' | grep "chr" >> $out_dir/RNAseq.depth.bed
elif [ $IFSTR == 2 ]; then
        echo the sequencing data is from antisense strand ...
        genomeCoverageBed -ibam $bam -split -dz -strand - | awk '{OFS="\t"}{print $1,$2,$2+1,$1":"$2,$3,"+"}' | grep "chr" > $out_dir/RNAseq.depth.bed
        genomeCoverageBed -ibam $bam -split -dz -strand + | awk '{OFS="\t"}{print $1,$2,$2+1,$1":"$2,$3,"-"}' | grep "chr" >> $out_dir/RNAseq.depth.bed
else
        echo the sequencing data is strandless ...
        genomeCoverageBed -ibam $bam -split -dz -strand + | awk '{OFS="\t"}{print $1,$2,$2+1,$1":"$2,$3,"+"}' | grep "chr" > $out_dir/RNAseq.depth.bed
		genomeCoverageBed -ibam $bam -split -dz -strand - | awk '{OFS="\t"}{print $1,$2,$2+1,$1":"$2,$3,"-"}' | grep "chr" >> $out_dir/RNAseq.depth.bed
fi
#date
#FIN

echo splitting files 4 parallelization ...
split -n l/$thread -d $out_dir/RNAseq.depth.bed $out_dir/RNAseq.depth.part.
rm -f $out_dir/RNAseq.depth.bed
#date
for eachp in $out_dir/RNAseq.depth.part*
do
	echo processing $eachp ...
#	date
	bash run1thread.sh $eachp $bed_reg $model $genome $out_dir &
	sleep 2s
#	date
done
wait
#date
echo merging the output cluster ...
cat $out_dir/RNAseq.depth.part*cluster | intersectBed -a - -b $bed_reg -s -wo | awk '{OFS="\t"}{print $1,$2,$3,$11,$4,$6,$7}' | sort -k 5 -r -n > $out_dir/combined.predicted.cluster
rm -f $out_dir/*.part*cluster
cat $out_dir/RNAseq.depth.part*.bedgraph > $out_dir/combined.raw.prediction
rm -f $out_dir/RNAseq.depth.part*.bedgraph









