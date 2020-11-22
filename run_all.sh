#!/bin/bash

dutchcorpus_path="/home/jer/Desktop/thesis-IK/DutchWebCorpus/sents/"
wablieft_path="/home/jer/Desktop/thesis-IK/wablieft/sents"
output_path="/home/jer/Desktop/UnsupNTS-Dutch/data/"

tmpfile="sents.txt"
dutchcorpusoutput="dutchcorpus-cleaned-sents.txt"
wablieftoutput="wablieft-sents.txt"

find $dutchcorpus_path -name "*.sents" -print | xargs cat >> $output_path$tmpfile
sed 's/.*|//' $output_path$tmpfile > $output_path$dutchcorpusoutput

rm $output_path$tmpfile

find $wablieft_path -name "*.sent" -print | xargs cat >> $output_path$wablieftoutput

python3 preprocess-wablieft.py

python3 nlwiktionary_parser.py

