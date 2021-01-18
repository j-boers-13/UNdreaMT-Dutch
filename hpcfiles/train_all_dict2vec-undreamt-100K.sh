#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-dict2vec-undreamt-100K
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeroen.boers1@gmail.com
#SBATCH --output=job-%j.log

module load Python
module load CUDA
head -n 100000 /scratch/s2748150/data/data/dutchcorpus-cleaned-sents.txt > /scratch/s2748150/data/data/dutchcorpus-cleaned-sents-100000.txt
head -n 100000 /scratch/s2748150/data/data/wablieft-cleaned-sents.txt > /scratch/s2748150/data/data/wablieft-cleaned-sents-100000.txt

singularity exec --bind /scratch/s2748150/data/data:/home/s2748150/data --nv pytorch_gpu.simg python3 UnsupNTS/undreamt/train.py --src data/dutchcorpus-cleaned-sents-100000.txt --trg data/wablieft-cleaned-sents-100000.txt --src_embeddings data/cmbembed.vec --trg_embeddings data/cmbembed.vec --save data/dict2vec.100K.undreamt --cuda --cutoff 0 --embedding_size 0 --dropout 0.3 --param_init 0.1 --iterations 300000 --max_sentence_length 50 --unsup
