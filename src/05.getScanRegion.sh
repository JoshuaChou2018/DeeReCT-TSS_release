#!/bin/bash
:<<USE
get the scanning regions from a bam file 
USE

function usage () {
        echo
        echo
        echo Usage:getScanRegion.sh  [bam_file] [prefix/label] [0/1/2 ifstranded] [bed region of interest]
        echo 0 means stranless 
        echo 1 means sense strand 
        echo 2 means antisense strand
        echo region of interest. i.e. protein coding gene/ all genes / intergenic  
        echo
}

if [ -z $1 ]; then
        usage
        exit
fi

BAM=${1}
LABEL=${2}
IFSTR=${3}
BED=${4}
echo bam file is $BAM
if [ $IFSTR == 1 ]; then
	echo sense 
	genomeCoverageBed -ibam $BAM -split -bg -strand + > $LABEL.RNAseq.plus.bedgraph
	genomeCoverageBed -ibam $BAM -split -bg -strand - > $LABEL.RNAseq.minus.bedgraph
	awk '{OFS="\t"}{print $1,$2,$3,$1":"$2":"$3,$4,"+"}' $LABEL.RNAseq.plus.bedgraph > $LABEL.RNAseq.bedgraph.bed
	awk '{OFS="\t"}{print $1,$2,$3,$1":"$2":"$3,$4,"-"}' $LABEL.RNAseq.minus.bedgraph >> $LABEL.RNAseq.bedgraph.bed
	intersectBed -a $LABEL.RNAseq.bedgraph.bed -b $BED -wo -s |  awk '{OFS="\t"}{print $1,$2 - 1000,$3 + 1000,$10,$5,$12}' | awk '$2 > 0 && $5 > 1'| sort | uniq | sortBed -i - > $LABEL.RNAseq.bedgraph.sorted.bed
elif [ $IFSTR == 2]; then
	echo antisense
	genomeCoverageBed -ibam $BAM -split -bg -strand - > $LABEL.RNAseq.plus.bedgraph
	genomeCoverageBed -ibam $BAM -split -bg -strand + > $LABEL.RNAseq.minus.bedgraph
	awk '{OFS="\t"}{print $1,$2,$3,$1":"$2":"$3,$4,"+"}' $LABEL.RNAseq.plus.bedgraph > $LABEL.RNAseq.bedgraph.bed
	awk '{OFS="\t"}{print $1,$2,$3,$1":"$2":"$3,$4,"-"}' $LABEL.RNAseq.minus.bedgraph >> $LABEL.RNAseq.bedgraph.bed
	intersectBed -a $LABEL.RNAseq.bedgraph.bed -b $BED -wo -S |  awk '{OFS="\t"}{print $1,$2 - 1000,$3 + 1000,$10,$5,$12}' | awk '$2 > 0 && $5 > 1'| sort | uniq | sortBed -i - > $LABEL.RNAseq.bedgraph.sorted.bed
else
	echo strandless
	genomeCoverageBed -ibam $BAM -split -bg > $LABEL.RNAseq.bedgraph
	awk '{OFS="\t"}{print $1,$2,$3,$1":"$2":"$3,$4,"+"}' $LABEL.RNAseq.bedgraph > $LABEL.RNAseq.bedgraph.bed
	intersectBed -a $LABEL.RNAseq.bedgraph.bed -b $BED -wo |  awk '{OFS="\t"}{print $1,$2 - 1000,$3 + 1000,$10,$5,$12}' | awk '$2 > 0 && $5 > 1'| sort | uniq | sortBed -i - > $LABEL.RNAseq.bedgraph.sorted.bed
fi
mergeBed -i $LABEL.RNAseq.bedgraph.sorted.bed -s -c 4,5,6 -o distinct,sum,distinct > $LABEL.RNAseq.bedgraph.merged.bed

