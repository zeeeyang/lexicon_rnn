#!/usr/bin/env bash
workspace=`pwd`
datadir=../data/
embdir=../../embeddings
lexicondir=../../lexicons
tooldir=../twitter_cnn/lexicon2/
function run()
{
   nohup $tooldir/$1 $embdir/glove.semeval.conj.pretrained.vec \
         $lexicondir/sentiment140.lex $datadir/train.fmt \
         $datadir/dev.fmt \
         $datadir/test.fmt \
         1>$workspace/$2.log 2>&1 &
}

run LexiconLambda lexicon.lambda.s140.1
run LexiconLambda lexicon.lambda.s140.2
run LexiconLambda lexicon.lambda.s140.3
