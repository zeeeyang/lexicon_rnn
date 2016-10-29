#!/usr/bin/env bash
#accept four params:
##    1. output_prefix
##    2. domain
##    3. lexicon
##    4. modelname

output_prefix=$1
domain=$2
lexicon=$3
modelname=$4
workspace=`pwd`
datadir=../data
tooldir=../../binary_task/binary_cnn/lexicon2/
embdir=../../embeddings
inputdir=../domain_data/$domain
outputdir=$workspace/${output_prefix}_$domain
mkdir -p $outputdir
rm -rf $outputdir/*

function run()
{
   $tooldir/$1 $embdir/glove.sentiment.conj.pretrained.vec\
   $lexicon\
   $datadir/sent+phrase.binary.clean.train\
   $datadir/raw.clean.train\
   $datadir/raw.clean.dev\
   $datadir/raw.clean.test\
   $2\
   $3\
   $4
}
files=`ls $inputdir/`
for file in $files; do
    echo $file
    inputfile=$inputdir/$file
    outputfile=$outputdir/$file
    run LexiconLambdaDecoder $modelname $inputfile $outputfile
done
cat $outputdir/* >> $outputdir/all
