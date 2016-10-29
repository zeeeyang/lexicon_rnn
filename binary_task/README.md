#Introduction
This fold contains code and data to replicate the experiments for the binary classification task on Stanford Sentiment Treebank.   
 + binary_cnn: source code folder. [binary_cnn/lexicon2/LexiconLambda.cc](binary_cnn/lexicon2/LexiconLambda.cc) is the main code.
 + data: data folder
 + exp: scripts folder 

#Usage
###Complie  
```
cd binary_cnn
./cmake.sh
```
  
###Run  
Assume you are in the *binary_task* folder.   

```
cd exp
```
  
Train with **SD-Lex**  
  
```
train_sd.sh
```

Train with **SWN-Lex**  
  
```
train_swn.sh
```
###Accuracy
You can use   
```
grep Accuracy *.log
```
  
to find out the classification accuracy. 
