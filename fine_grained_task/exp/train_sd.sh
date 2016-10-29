#!/usr/bin/env bash
workspace=`pwd`
datadir=../data/
tooldir=../five_cnn/lexicon2/
embdir=../../embeddings
lexicondir=../../lexicons
function run()
{
   nohup $tooldir/$1 $embdir/glove.sentiment.conj.pretrained.vec\
   $lexicondir/stanford.tree.lexicon\
   $datadir/sent+phrase.clean.train\
   $datadir/raw.clean.train\
   $datadir/raw.clean.dev\
   $datadir/raw.clean.test\
   1>$workspace/$2.log 2>&1 &
}

run LexiconLambda lexicon.five.lambda.new2.1
run LexiconLambda lexicon.five.lambda.new2.2
run LexiconLambda lexicon.five.lambda.new2.3
