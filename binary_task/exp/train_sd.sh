#!/usr/bin/env bash
workspace=`pwd`
datadir=../data
embdir=../../embeddings
lexicondir=../../lexicons
tooldir=../binary_cnn/lexicon2/
function run()
{
   nohup $tooldir/$1 $embdir/glove.sentiment.conj.pretrained.vec\
   $lexicondir/stanford.tree.lexicon\
   $datadir/sent+phrase.binary.clean.train\
   $datadir/raw.clean.train\
   $datadir/raw.clean.dev\
   $datadir/raw.clean.test\
   1>$workspace/$2.log 2>&1 &
}

run LexiconLambda lexicon.binary.lambada.sd.d0.5.3
run LexiconLambda lexicon.binary.lambada.sd.d0.5.4
