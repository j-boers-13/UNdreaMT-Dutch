#!/bin/bash

dutchcorpus_path="/home/jer/Desktop/thesis-IK/DutchWebCorpus/sents/"
wablieft_path="/home/jer/Desktop/thesis-IK/wablieft/sents"
output_path="/home/jer/Desktop/UnsupNTS-Dutch/data/"

src_mono_emb="src-emb-mono-default-cbow.txt"
trg_mono_emb="trg-emb-mono-default-cbow.txt"
src_mapped="src-emb-mapped-default-cbow.emb"
trg_mapped="trg-emb-mapped-default-cbow.emb"
combined_emb="cmbembed"

combined_emb_file="cmbembed.vec"

tmpfile="sents_tmp.txt"
tmpfile2="sents_tmp2.txt"

strong_pairs="strong_pairs.txt-K5.txt"
weak_pairs="weak_pairs.txt-K5.txt"

wabliefttmp="wablieft-sents.txt"
dutchcorpustmp="dutchcorpus-sents.txt"
testdutchcorpustmp="dutchcorpus-sents-test.txt"
dutchcorpusoutput="dutchcorpus-cleaned-sents.txt"
testdutchcorpusoutput="dutchcorpus-test.txt"
wablieftoutput="wablieft-cleaned-sents.txt"
combinedoutput="all-cleaned-sents.txt"

python3 nlwiktionary_parser.py

python3 woordenlijst_scraper.py

# Clean and split dutchcorpus
find $dutchcorpus_path -name "*.sents" -print | xargs cat >> $output_path$tmpfile
cat $output_path$tmpfile | sed -e 's/‘//g' -e 's/’//g' -e 's/.*|//g' -e 's/”//g' -e 's/„//g' -e 's/"//g' -e "s/'//g" -e "s/\s\{2,\}/ /g" -e "s/([^)]*)//g" > $output_path$tmpfile2
head -n 300000 $output_path$tmpfile2 > $output_path$dutchcorpustmp
tail -n 10000 $output_path$tmpfile2 > $output_path$testdutchcorpustmp

# Clean wablieft corpus
find $wablieft_path -name "*.sent" -print | xargs cat | \
sed -e 's/‘//g' -e 's/’//g' -e 's/.*|//g' -e 's/”//g' -e 's/„//g' -e 's/"//g' -e "s/'//g" -e "s/\s\{2,\}/ /g" -e "s/([^)]*)//g" > $output_path$wabliefttmp
python3 preprocess-corpora.py

# Remove tmp files
rm $output_path$wabliefttmp
rm $output_path$tmpfile
rm $output_path$tmpfile2
rm $output_path$dutchcorpustmp
rm $output_path$testdutchcorpustmp

head -n 257202 $output_path$dutchcorpusoutput > $output_path$dutchcorpustmp
cat $output_path$dutchcorpustmp > $output_path$dutchcorpusoutput
rm $output_path$dutchcorpustmp


#Create default settings word2vec embeddings (cbow)
python3 get_word2vecs.py

#Create UNdreaMT paper hyperparameter settings word2vec (cbow and skipgram)
python3 get_word2vecs_undreamt.py

#Create VecMap Cross-Lingual embeddings for three created src and trg embeddings
#Artetxe et al. suggests using default setup for vecmap
python3 vecmap/map_embeddings.py --unsupervised $output_path$src_mono_emb $output_path$trg_mono_emb $output_path$src_mapped $output_path$trg_mapped --cuda

python3 vecmap/map_embeddings.py --unsupervised "${output_path}src-emb-mono-undreamt-cbow.txt" "${output_path}trg-emb-mono-undreamt-cbow.txt" "${output_path}src-mapped-undreamt-cbow.emb" "${output_path}trg-mapped-undreamt-cbow.emb" --cuda

python3 vecmap/map_embeddings.py --unsupervised "${output_path}src-emb-mono-undreamt-skipgram.txt" "${output_path}trg-emb-mono-undreamt-skipgram.txt" "${output_path}src-mapped-undreamt-skipgram.emb" "${output_path}trg-mapped-undreamt-skipgram.emb" --cuda

# Theoretical training example, actual training done by batch files on supercomputer
python3 "UnsupNTS/undreamt/train.py" --src "$output_path$dutchcorpusoutput" --trg "$output_path$wablieftoutput" \
 --src_embeddings "$output_path$src_mapped" --trg_embeddings "$output_path$trg_mapped"  --save "model/first-model" \
--batch 36 --cuda --unsup

# Generate strong and weak pairs from definitions
python3 generate_pairs.py

#Create dict2vec embeddings
./dict2vec/dict2vec -input "$output_path$combinedoutput" -output "$output_path$combined_emb" -strong-file "$output_path$strong_pairs" -weak-file "$output_path$weak_pairs" -size 300 -window 5 -threads 8 -sample 1e-4 -min-count 5 -negative 5 -strong-draws 4 -beta-strong 0.8 -weak-draws 5 -beta-weak 0.45 -alpha 0.025 -threads 8 -epoch 5 -save-each-epoch 0

python3 "UnsupNTS/undreamt/train.py" --src "$output_path$dutchcorpusoutput" --trg "$output_path$wablieftoutput"  --src_embeddings "$output_path$combined_emb_file" --trg_embeddings "$output_path$combined_emb_file"  --save "model/dict2vec-all" --batch 36 --cuda --unsup --start_save 18000 --stop_save 24000

exit
