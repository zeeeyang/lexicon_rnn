#Introduction
This fold contains code and data to replicate the experiments for the binary classification task on SemEval 2013 Twitter dataset.   
 + twitter_cnn: source code folder. [twitter_cnn/lexicon2/LexiconLambda.cc](twitter_cnn/lexicon2/LexiconLambda.cc) is the main code.
 + data: data folder
 + exp: scripts folder 

#Usage
###Complie  
```
cd twitter_cnn
./cmake.sh
```

###Run  
Assume you are in the *twitter_task* folder. 

```
cd exp
```

Train with **TS-Lex**

```
train_dyt.sh
```

Train with **S140-Lex**

```
train_s140.sh
```
###Accuracy
You can use 
```
grep Accuracy *.log
```

to find out the classification accuracy. 
