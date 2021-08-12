#!/bin/bash
:<<USE
do clustering on output from deep_TSS, calulate empirical p value and estimate FDR...
USE


function usage () {
        echo
        echo
        echo Usage:clusterTSS.sh [deep_out] [prefix/label] [score cutoff] [distance cutoff] [prop]
        echo deep_out is the output from deep learning model. 
        echo score cutoff, default is 0.1, any score less than this cutoff will be counted as 0 
        echo distance cutoff, default is 10bp, any locus within this distance will be group into one cluster 
		echo prop is the probabilty of a site with prediction score s, can not be a true TSS 
        echo 
        echo
}

if [ -z $2 ]; then
        usage
        exit
fi
deep_out=${1}
label=${2}
# default prediction score cut 1 
scut=0.1
# default distance to merge TSS into cluster is 10nt 
dcut=10
# default proability of a site not to be a TSS is 0.971 with score 0.1 
prop=0.971
if [ -z $3 ]; then
	scut=$3
fi

if [ -z $4 ]; then
	dcut=$4
fi

echo change to 0-based bed format 
if [ ! -d tmp ]; then
	mkdir tmp
fi

#
echo change to 0-based bed format 
awk -v scut=$scut '{OFS="\t"}{if($4 > scut) print $1,$2 - 1,$3,$6,$4,$5}' $deep_out | grep -v "chrM" > ./tmp/$label.scut$scut.bed
grep "chr1" ./tmp/$label.scut$scut.bed | sortBed -i - > ./tmp/$label.scut$scut.sorted.bed
grep "chr[2-9XY]" ./tmp/$label.scut$scut.bed | sortBed -i - >> ./tmp/$label.scut$scut.sorted.bed

echo clustering TSS 
mergeBed -i ./tmp/$label.scut$scut.sorted.bed -s -d $dcut -c 4,5,6 -o first,sum,distinct > ./tmp/$label.scut$scut.dcut$dcut.merged.sum.bed
#tn=`wc -l ./tmp/$label.scut$scut.dcut$dcut.merged.sum.bed`
# 0.971 is the probability of a tss to be 
awk -v prop=$prop '{OFS="\t"}{print $1,int(($2+$3)/2),int(($2+$3)/2) + 1,$5,$4,$6,prop^int($5)}' ./tmp/$label.scut0.1.dcut10.merged.sum.bed | awk '$5 > 1' | sort -k 4 -r -n  > $label.predicted.TSS.withpval
# clear tmporary files 
rm -f ./tmp/$label.*








