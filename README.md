#Lexicon RNN
**lexicon_rnn** is the code for my EMNLP 2016 paper [Context-Sensitive Lexicon Features for Neural Sentiment Analysis](https://aclweb.org/anthology/D16-1169). 
It contains a context-sensitive **lexicon-based method** based on a simple weighted-sum model, using a  **recurrent  neural  network**  to  learn  the  sentiments strength,  intensification and negation of  lexicon  sentiments  in  composing  the  sentiment value of sentences.  
Using this toolkits, you probably can:  
 1. Integrate sentiment lexicons into recurrent neural network models.  
 2. Obtain competitive performance on standard sentiment classification benchmarks for both in-domain and cross-domain datasets. More specifically, we use the Stanford Sentiment Treebank, the SemEval 2013 Twitter sentiment classification dataset and  the mixed domain dataset of product reviews (T{\"a}ckstr{\"o}m, Oscar and McDonald, Ryan, 2011) in our paper.  
 3. Produce potentially interpretable sentiment composition details, such as intensification and negation.   
 4. Use filtered embeddings and processed lexicons invovled in our paper.   
  
The twitter dataset is available by an email request to the first author. 
#Usage
Please see README.md in every task directory. 
#Lexicons 
We provide four preprocessed sentiment lexicons used in our paper, including [TS-Lex](./lexicons/sspe.lex2), [S140-Lex](./lexicons/sentiment140.lex), [SD-Lex](./lexicons/stanford.tree.lexicon) and [SWN-Lex](./lexicons/sentiwordnet.lex). 
#Embeddings
We construct our pretrained embedding tables from [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip).  
[glove.sentiment.conj.pretrained.vec](./embeddings/glove.sentiment.conj.pretrained.vec) and [glove.sentiment.conj.pretrained.vec](./embeddings/glove.sentiment.conj.pretrained.vec) are embedding tables for the SST and SemEval dataset, respectively. 
#Citation
---
If you found our codes and preprocessed data are useful for your research, please cite

    @InProceedings{teng-vo-zhang:2016:EMNLP2016,
    author    = {Teng, Zhiyang  and  Vo, Duy Tin  and  Zhang, Yue},
    title     = {Context-Sensitive Lexicon Features for Neural Sentiment Analysis},
    booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
    month     = {November},
    year      = {2016},
    address   = {Austin, Texas},
    publisher = {Association for Computational Linguistics},
    pages     = {1629--1638},
     url       = {https://aclweb.org/anthology/D16-1169}
    }
