#!/bin/bash
#Example run:
#bash auto.sh thaliana_tata+.fa 4800 1000 600 1 20 1000
pipeline='../pipeline'
sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python $pipeline/split.py $sdir/$1 0.9 $2 $4
python $pipeline/shorten.py $sdir/lp_1 $2 $4 $3

#python $sdir/extend.py $sdir/p_1

iter=$6
for ((i=1;i<$iter+1;i++))
do 
  python $pipeline/train_model.py $pipeline/params.txt model_$i
  python $pipeline/predictor.py $sdir/model_$i $sdir/lp_2 $4 1 $sdir/output.txt
  java -jar $pipeline/draw.jar -setp $sdir/output.txt -out $sdir/pics_$i -count 12 -step 1 -maxf 10
  echo $i >> 1res.txt
  java -jar $pipeline/evaluate.jar -setp $sdir/output.txt -pos $2 >> 1res.txt
  if [ $i -lt  $iter ]; then   	
    python $pipeline/nboost.py $sdir/model_$i $sdir/lp_1 $4 1 $2 $3 $5 $sdir/new_neg.txt $7
    python $pipeline/merge.py $sdir/n_1 $sdir/new_neg.txt $sdir/n_1
  fi
done
