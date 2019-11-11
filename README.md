# OffensEval19-SVM
This repository contains the (mainly hard-coded) code for the SVM used in the OffensEval task 2019.
For a detailed decription see: [TuKaSt at SemEval-2019 Task 6: something old, something neu(ral): Traditional and neural approaches to offensive text classification.](https://www.aclweb.org/anthology/S19-2134.pdf) 

### Usage
python svm.py --vector {1,2,3} --model {A,B,C} training_file.tsv  prediction_infile.tsv  outfile  --tenfold
  - vector: Chose Vector representation used in the model:
    - 1: combined_fixed 
    - 2: combined_positioned 
    - 3: emb_only 
    - default is combined_fixed
  - model: Use Model to be trained:
    - A: Subtask A Offensive/Not-Offensive
    - B: Subtask B Targeted/Untargeted 
    - C: Subtask C Targeted towards Individual/Group/Other
    - default is A
    
When used without the tenfold flag, the SVM is trained on the specified training_file.tsv, predicts the items in the prediction_infile.tsv and writes the predictions to the outfile. When the tenfold flag is used, the SVM is ten-fold evaluated on the training_file.tsv.
A binary fastext model file (../cc.en.300.bin) and vader lexicon (../vader.txt) should be placed in the parent directory of the python project folder.

