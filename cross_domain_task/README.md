#Introduction
This fold contains code and data to replicate the cross-domain experiments. 
[binary_cnn/lexicon2/LexiconLambdaDecoder.cc](../binary_task/binary_cnn/lexicon2/LexiconLambda.cc) is the main code.  
 + data: data folder
 + exp: scripts folder 

#Usage
###Set Model Paths
please change the model path in exp [run_decode_main.sh](./exp/run_decode_main.sh).
```
##This variable keep the model you generated in the binary task with SD-Lex. 
sd_model=
##This variable keep the model you generated in the binary task with SWN-Lex. 
swn_model=
```
###Run  
```
cd exp
nohup ./run_decode_main.sh 1>decode.log 2>&1 & 
```

###Results
You can find three kinds of folders. 
 + sd.v2.sd\*: use SD-Lex both to train and test
 + sd.v2.swn\*: use SD-Lex to train, and use SWN-Lex to test 
 + swn.v2.swn\*: use SWN-Lex both to train and test

Show accuracy:
```
tail -n 100 decode.log
```
