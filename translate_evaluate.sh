set -e

#defining paths
ts=`pwd`
src=en
tgt=sen
tsdata=data
codepath=UnsupNTS/undreamt
ntsevalcode=UnsupNTS/utils/evaluate.py
gendir=data/predictions
genfile=$gendir/gen_lower.$tgt
model=model
logdir=data/logs/TS.GEN
file=dutchcorpus-test
srcfile=data/$file.txt

#creating new directories
mkdir -p "$gendir"
mkdir -p "$model"
mkdir -p "$logdir"



#Generating simplifications
nlines=( final )
control_nums=( 1 )
for ncontrol in "${control_nums[@]}"
do
	for nline in "${nlines[@]}"
	do
	 	modelnum=$nline
    	pref="first-model"
		noise=0.0
		pref="$pref"
		modelfile=$model/$pref.$modelnum.src2trg.pth
		echo $pref.it$modelnum.src2trg.pth
		python3 -u "$codepath/translate.py" "$modelfile" --input "$srcfile" --output "$genfile.src2trg.${pref}.$nline.$file"  --noise $noise \
		--batch_size 50 --ncontrol $ncontrol \
		>> "$logdir/out.src2trg.$pref" 
		
	done 
done






#Evaluating the Simplifications
nlines=( final )
control_nums=( 1 )
for ncontrol in "${control_nums[@]}"
do
	for nline in "${nlines[@]}"
	do
    	pref="first-model"
		noise=0.0
	 	modelnum=$nline
		modelfile=$model/$pref.$modelnum.src2trg.pth
		echo $pref.it$modelnum.src2trg.pth
		genf=$genfile.src2trg.${pref}.$modelnum.$file
		python3 UnsupNTS/predictions/noredund.py < "$genf" > "${genf}.noredund"
		genf="$genf.noredund"
		python UnsupNTS/utils/lev.py --input "$genf" --source "$srcfile" 
		python UnsupNTS/utils/fk_ts_ds.py -i "$genf" -src "$srcfile"
	done 
done
