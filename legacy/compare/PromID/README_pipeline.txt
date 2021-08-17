Draw.jar
-------------------------------------------------------------------------------------------------
Draws scoring landscapes given input from PromID
-------------------------------------------------------------------------------------------------
java -jar draw.jar -setp input.txt -out pics -count 12 -step 1 -maxf 10

-setp: predictions from PromID
-out: output directory;
-w: width of one landscape
-h: height of one landscape
-count: number of graphs per picture
-maxf: maximum number of generated files (default 10)


Evaluate.jar
-------------------------------------------------------------------------------------------------
Outputs performance metrics for input from PromID
-------------------------------------------------------------------------------------------------
java -jar evaluate.jar -setp input.txt -pos 4800 -mr 500 -md 1000
-setp: predictions from PromID
-pos: true TSS position
-mr: margin for error (default 500)
-md: minimum distance between promoters (default 1000)


split.py
-------------------------------------------------------------------------------------------------
Extract positive sequences (Long and Short) from the training set with long sequences
-------------------------------------------------------------------------------------------------
python split.py input.fasta 0.9 4800 600
Parameter 1: input training set
Parameter 2: size of training positive set
Parameter 3: true TSS positions
Parameter 4: size of input for TensorFlow model/Sliding window size

shorten.py
-------------------------------------------------------------------------------------------------
Extract initial negative sequences randomly from the training set with long sequences
Size is always equal to number of positive sequences
-------------------------------------------------------------------------------------------------
python shorten.py lp_1 4800 600
Parameter 1: input training set with long sequences (output of split.py)
Parameter 2: true TSS positions
Parameter 3: size of input for TensorFlow model/Sliding window size


train_model.py
-------------------------------------------------------------------------------------------------
Trains TensorFlow models
-------------------------------------------------------------------------------------------------
python train_model.py params.txt model_1
Parameter 1: text file with neural network parameters
Parameter 2: name used for saving the new model 


predictor.py
-------------------------------------------------------------------------------------------------
Makes predictions on long sequences using sliding window
-------------------------------------------------------------------------------------------------
python predictor.py model_1 lp 600 1 output.txt
Parameter 1: TensorFlow model to use
Parameter 2: file with long sequences
Parameter 3: size of the sliding window/model input
Parameter 4: step size
Parameter 5: output file


nboost.py
-------------------------------------------------------------------------------------------------
Choose new negatives from long sequences
-------------------------------------------------------------------------------------------------
python nboost.py model_1 lp 600 1 4800 500 2 new_neg.txt
Parameter 1: TensorFlow model to use
Parameter 2: file with long sequences
Parameter 3: size of the sliding window/model input
Parameter 4: step size
Parameter 5: true TSS position
Parameter 6: minimum distance between new negative sequences
Parameter 7: number of new negatives to pick from one sequences
Parameter 8: output file

merge.py
-------------------------------------------------------------------------------------------------
Merge new negatives with old negatives
-------------------------------------------------------------------------------------------------
python merge.py n_1 new_neg.txt n_1
Parameter 1: old negatives file
Parameter 2: new negatives file
Parameter 3: output (often same as old negatives)