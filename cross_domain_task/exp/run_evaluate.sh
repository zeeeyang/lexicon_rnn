#!/bin/bash
domains="books dvds electronics music videogames"
prefix=$1
rm -rf ${prefix}.all
for domain in $domains; do
    echo $domain
    python ./count_acc.py ${prefix}_${domain}/all
    cat ${prefix}_${domain}/all >> ${prefix}.all
done
echo "Results of "$prefix
python ./count_acc.py ${prefix}.all

