#!/bin/bash
:<<USE
USE

function usage () {
        echo
        echo
		echo Usage:run1thread.sh [depth_bed] [bed_regions] [model] [genome_fa] [output_directory]
        echo
		echo
}

if [ -z $5 ]; then
        usage
        exit
fi

depth_bed=${1}
bed_reg=${2}
model=${3}
genome=${4}
out_dir=${5}

grep "+" $depth_bed | intersectBed -a - -b $bed_reg -s -u | cut -f 1,2,5 > $depth_bed.plus
grep "-" $depth_bed | intersectBed -a - -b $bed_reg -s -u | cut -f 1,2,5 > $depth_bed.minus

echo extracting genomic regions 4 scanning ...
# only keep the loci with depth > 1 read 
awk '$5 > 1' $depth_bed | intersectBed -a - -b $bed_reg -s -wo | awk '{OFS="\t"}{print $1,$2 - 1000,$3 + 1000,$10,$5,$12}' | awk '$2 > 0' | sort | uniq | sortBed -i - > $depth_bed.sorted
mergeBed -i $depth_bed.sorted -s -c 4,5,6 -o distinct,sum,distinct | awk '$2 > 500' > $depth_bed.merged
### 
echo scanning the genome ...
python src/05.scan.app.v2.py $depth_bed.merged $out_dir $model $depth_bed.plus $depth_bed.minus $genome 5 0 0.1 1>$depth_bed.scan.log 2>$depth_bed.scan.elog

echo cleaning tmporary files 
rm -f $depth_bed
rm -f $depth_bed.plus
rm -f $depth_bed.minus
rm -f $depth_bed.sorted
rm -f $depth_bed.coverage_data.p

scut=0.1
dcut=10
prop=0.971

echo clustering the predictions ...
echo change to 0-based bed format 
awk -v scut=$scut '{OFS="\t"}{if($4 > scut) print $1,$2 - 1,$3,"c",$4,$5}' $depth_bed.merged.bedgraph \
	 | grep -v "chrM" | sortBed -i | mergeBed -i - -s -d $dcut -c 4,5,6 -o first,sum,distinct > $depth_bed.rawout
awk -v prop=$prop '{OFS="\t"}{print $1,int(($2+$3)/2),int(($2+$3)/2) + 1,$5,$4,$6,prop^int($5)}' $depth_bed.rawout | awk '$5 > 1' > $depth_bed.cluster
rm -f $depth_bed.rawout
#rm -f $depth.mergeBed.bedgraph




