set -e

#defining paths
modeldir=/home/jer/Desktop/hpcfiles/newest-models/
codepath=/home/jer/Desktop/hpcfiles/home/UnsupNTS/undreamt
ntsevalcode=/homme/jer/Desktop/hpcfiles/home/UnsupNTS/utils/evaluate.py
gendir=/home/jer/Desktop/hpcfiles/predictions
genfile=$gendir/gen_lower
testfile=/home/jer/Desktop/hpcfiles/data/dutchcorpus-test-cleaned.txt
levenshteinfile=/home/jer/Desktop/hpcfiles/data/levenshtein.txt
fktsdsfile=/home/jer/Desktop/hpcfiles/data/fk_ts_ds.txt

srcfile=/home/jer/Desktop/hpcfiles/data/src-file.txt

modelnames=( \
"dict2vec.50K.undreamt.final.src2trg.pth" \
"dict2vec.100K.undreamt.final.src2trg.pth" \
"dict2vec.all.undreamt.final.src2trg.pth" \
"word2vec.default.cbow.final.src2trg.pth" \
"word2vec.undreamt.cbow.final.src2trg.pth" \
"word2vec.undreamt.skipgram.final.src2trg.pth" \
)

#Evaluating the Simplifications

head -n 500 $testfile > $srcfile

for modelname in ${modelnames[@]}; do
	noise=0.0
	modelfile=$modeldir/$modelname
	echo $modelname
	python3 -u "$codepath/translate.py" "$modelfile" --input "$srcfile" --output "$genfile.translation.$modelname"  --noise $noise \
	--batch_size 100 \
	>> "$modeldir/log.$modelname"
done

for modelname in ${modelnames[@]}; do
	noise=0.0
	modelfile=$modeldir/$modelname
	echo $modelname
	genf=$genfile.translation.$modelname
	python3 /home/jer/Desktop/hpcfiles/home/UnsupNTS/predictions/noredund.py < "$genf" > "${genf}.noredund"
	genf="$genf.noredund"
	python /home/jer/Desktop/hpcfiles/home/UnsupNTS/utils/lev.py --input "$genf" --source "$srcfile" >> levenshteinfile
	python /home/jer/Desktop/hpcfiles/home/UnsupNTS/utils/fk_ts_ds.py -i "$genf" -src "$srcfile" >> fktsdsfile
done