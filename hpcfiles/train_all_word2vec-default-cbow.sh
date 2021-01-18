#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-default-word2vec-cbow
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeroen.boers1@gmail.com
#SBATCH --output=job-%j.log

module load Python
module load CUDA
singularity exec --bind /scratch/s2748150/data/data:/home/s2748150/data --nv pytorch_gpu.simg python3 UnsupNTS/undreamt/train.py --src data/dutchcorpus-cleaned-sents.txt --trg data/wablieft-cleaned-sents.txt --src_embeddings data/src-emb-mapped-default-cbow.emb --trg_embeddings data/trg-emb-mapped-default-cbow.emb --save data/word2vec.default.cbow --cuda --cutoff 0 --embedding_size 0 --dropout 0.3 --param_init 0.1 --iterations 300000 --max_sentence_length 50 --unsup
