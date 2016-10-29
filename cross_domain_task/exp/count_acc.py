#!/usr/bin/env python
# coding=utf-8
import sys

if len(sys.argv) !=2:
    print 'Usage: %s input' %sys.argv[0]
    sys.exit(0)

input_file = open(sys.argv[1], "r")
acc = 0.0
total = 0
for line in input_file:
    gold, neg, pos, score = line.strip().split()
    total +=1
    gold = int(gold)
    neg = int(neg)
    pos = int(pos)
    score = float(score)
    if neg >pos and gold == 0:
        acc += 1
    elif neg <= pos and gold == 1:
        acc += 1
    #else:
    #    print line,
print "Accuracy: ", acc/total, acc, total
