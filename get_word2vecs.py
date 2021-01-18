import word2vec

SRC_MONO="data/dutchcorpus-cleaned-sents.txt"
TRG_MONO="data/wablieft-cleaned-sents.txt"

def main():
    word2vec.word2vec(SRC_MONO, 'data/src-emb-mono-default-cbow.txt', binary=False, verbose=True, threads=8, size=200, window=8, negative=25, hs=0 , sample=1e-4)
    word2vec.word2vec(TRG_MONO, 'data/trg-emb-mono-default-cbow.txt', binary=False, verbose=True, threads=8, size=200, window=8, negative=25, hs=0 , sample=1e-4)

if __name__ == '__main__':
    main()