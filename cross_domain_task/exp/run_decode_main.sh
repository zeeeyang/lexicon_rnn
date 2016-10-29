#!/bin/bash
domains="books dvds electronics music videogames"
#set your lexicon paths here
sd_lex=../../lexicons/stanford.tree.lexicon
swn_lex=../../lexicons/sentiwordnet.lex
#set your model paths here
sd_model=
swn_model=
#default output directory prefix
sd_sd_prefix="sd.v2.sd"
sd_swn_prefix="sd.v2.swn"
swn_swn_prefix="swn.v2.swn"
i=0
for domain in $domains; do
    echo $domain
    ./decode_lexicon.sh ${sd_sd_prefix} $domain ${sd_lex} ${sd_model} & 
    ./decode_lexicon.sh ${sd_swn_prefix} $domain ${swn_lex} ${sd_model} & 
    ./decode_lexicon.sh ${swn_swn_prefix} $domain ${swn_lex} ${swn_model} & 

    #you can enable code below to use more processes to decode

    ##let "i=i+1"
    ##if [ $i -eq 3 ]; then
    ##    echo "wait"
    ##    let "i=0"
    ##    wait
    ##fi
    #

    wait
done
wait
./run_evaluate.sh ${sd_sd_prefix}
./run_evaluate.sh ${sd_swn_prefix}
./run_evaluate.sh ${swn_swn_prefix}
