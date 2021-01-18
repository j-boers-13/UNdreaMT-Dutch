import word2vec

SRC_MONO="data/dutchcorpus-cleaned-sents.txt"
TRG_MONO="data/wablieft-cleaned-sents.txt"

## Word2vec's with the hyperparameters described in Artetxe et al. 2018

def main():
    word2vec.word2vec(SRC_MONO, 'data/src-emb-mono-undreamt-cbow.txt', binary=False, verbose=True, threads=8, size=300, window=10, negative=10, sample=1e-5)
    word2vec.word2vec(TRG_MONO, 'data/trg-emb-mono-undreamt-cbow.txt', binary=False, verbose=True, threads=8, size=300, window=10, negative=10, sample=1e-5)

    word2vec.word2vec(SRC_MONO, 'data/src-emb-mono-undreamt-skipgram.txt', binary=False, verbose=True, threads=8, size=300, window=10, negative=10, sample=1e-5, cbow=False)
    word2vec.word2vec(TRG_MONO, 'data/trg-emb-mono-undreamt-skipgram.txt', binary=False, verbose=True, threads=8, size=300, window=10, negative=10, sample=1e-5, cbow=False)

if __name__ == '__main__':
    main()