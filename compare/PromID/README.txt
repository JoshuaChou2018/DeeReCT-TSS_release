Training requires TensorFlow and sklearn to be installed. I used TensorFlow 1.3 on Ubuntu 14. 
Variable pipeline in auto.sh needs to point to the location of pipeline folder.

Example run:
bash auto.sh input.fasta 4800 500 251 1 10 5000

Parameters:
1 - input file with long sequences
2 - start of the promoter region
3 - minimum distance away from promoter region to pick negative sequences for training/testing
4 - promoter region length 
5 - how many new negatives to pick from 1 sequence (1 is recommended)
6 - number of iterations
7 - maximum number of new negatives to pick on one iteration

During the training, after each iteration results will be printed in 1.res file.
If the results are not improving the run can be terminated. 

input.fasta contain long [-5000 +5000] sequences with TATA+ TSS at position +1. 

After 1 model or 2 models (TATA+ and TATA-) are generated they should be put into folder with size of the window length (for example 251) and that folder should be in the same folder as PromFind jar file (PromFind_Dist). The models should be renamed to model_1 (TATA-) and model_2 (TATA+). After that they are ready to be used by PromFind for TSS prediction. 


PromRegion - build graph showing importance of different regions inside promoter.
Example run:
java -jar pipeline/PromRegion.jar -mod ../PromFind/1500/model_BOTH_1500 -set ../PromFind/TATA-.txt_2 -out ../PromFind/test111.png
	-set: file with long sequences
	-mod: location of trained model
	-out: output