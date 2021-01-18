import json
from collections import defaultdict
import pickle
import sys
import os.path
import argparse
import numpy as np
import time
from numpy.linalg import norm
from collections import Counter
import re
import nltk

WIKTIONARY_PATH="data/wiktionary_definitions.p"
WL_SYN_PATH="data/synonyms2.p"
WL_DEF_PATH="data/definitions2.p"
JSON_PATH="data/synonyms.json"
DEF_DICT = defaultdict(set)

def cosineSim(v1, v2):
    """Return the cosine similarity between v1 and v2 (numpy arrays)"""
    dot_prod = np.dot(v1, v2)
    return dot_prod / (norm(v1) * norm(v2))


def loadEmbedding(filename, list_words):
    """
    Read the file <filename> and generate the embedding matrix. Only load
    embeddings of words in <list_words>. There is no reason to load the
    embedding of a word if we are not going to do computation with it.
    """

    # Read the file to get the number of words and the embedding dimension
    print("   Reading \"{}\" to get the dimension and the number of"
          " words ... ".format(filename), end="")
    nb_word = 0

    with open(filename) as f:
        first_line = f.readline().split()
        _, nb_dims = first_line[0], int(first_line[1])

        # for each line, add 1 to nb_word if word is in list_words
        for line in f:
            if line.split()[0] in list_words:
                nb_word += 1

    print("Done.\n   Loading {} embeddings of dimension"
          " {} ... ".format(nb_word, nb_dims), end="")

    # each row is the embedding of a word
    embedding = np.zeros((nb_word, nb_dims))

    # dictionaries to map each word and their respective index
    numToWords, wordsToNum = {}, {}

    # Read again the file to extract embeddings and put them into the matrix
    with open(filename) as f:
        first_line = f.readline()

        idx = 0
        for line in f:
            line = line.split()
            word, vals = line[0], list(map(float, line[1:]))

            # add embedding only if the word is in list_words
            if word in list_words:
                embedding[idx] = vals
                numToWords[idx] = word
                wordsToNum[word] = idx
                idx += 1

    print("Done.")
    print("   Normalizing the embeddings ... ", end="")

    # norm(., axis=1) gives the norm of each rows. It is an array with
    # dimension (n, ). To divide each coefficicent of embedding with the
    # corresponding norm, we need to reshape the array to (n, 1)
    embedding = embedding / norm(embedding, axis=1)[:, np.newaxis]

    print("Done.")

    return embedding, numToWords, wordsToNum

def add_syn_defs_from_json(stopwords):
    syn_dict = defaultdict(set)

    with open(JSON_PATH) as f:
        data = json.load(f)
        for synset in data:
            word = synset['base']
            synonyms = synset['list']
            for syn in synonyms:
                synlower = syn.lower()
                synlower = set([re.sub(r'[\W]+', "", x) for x in synlower.split()])
                synlower.discard("")
                synlower.difference_update(stopwords)
                if len(synlower) == 0:
                    continue
                elif len(synlower) == 1:
                    syn_dict[word].update(synlower)
                else:
                    syn_dict[word].update(synlower)
    return syn_dict

def add_woordenlijst_pickles(syn_dict, def_dict, stopwords):
    with open(WL_SYN_PATH, "rb") as synonyms:
        wl_syn_dict = pickle.load(synonyms)

    for key, value in wl_syn_dict.items():
        if len(value) != 0:
            synlower = set([re.sub(r'[\W]+', "", x).lower() for x in value])
            synlower.discard("")
            synlower.difference_update(stopwords)
            syn_dict[key].update(synlower)

    with open(WL_DEF_PATH, "rb") as definitions:
        wl_def_dict = pickle.load(definitions)

    for key, value in wl_def_dict.items():
        if len(value) != 0:
            deflower = set([re.sub(r'[\W]+', "", x).lower() for x in value])
            deflower.discard("")
            deflower.difference_update(stopwords)
            if len(deflower) != 0:
                def_dict[key].update(deflower)


def add_wiktionary_defs():
    with open("data/wiktionary_definitions.p", "rb") as wikt_defs:
        return pickle.load(wikt_defs)

def generate_pairs(def_dict, syn_dict, embedding_fn, strg_fn, weak_fn, K):
    """
    Generate weak and strong pairs of words based on definitions in
    defs_fn. A and B are a strong pair if :
        - A is in definition of B
        - B is in definition of A
    All others pairs of words (ie a word and a word from its definition)
    are considered as a weak pair.

    """
    uniq_words = set() # list of all words in definition_file

    for words in def_dict.values():
        if words != set():
            uniq_words.update(words)
    for words in syn_dict.values():
        if words != set():
            uniq_words.update(words)

    print("Done.")
    print("   Entries in \"{}\":\t{}".format("def dict", len(def_dict)))
    print("   Entries in \"{}\":\t{}".format("syn dict", len(syn_dict)))

    print("   Uniq words in \"{}\":\t{}".format("both", len(uniq_words)))

    # load pre-existing embeddings.
    print("\n-- Loading embedding from \"{}\"".format(embedding_fn))
    embedding, numToWords, wordsToNum = loadEmbedding(embedding_fn, uniq_words)


    # generate strong and weak pairs
    print("\n-- Generating strong and weak pairs")
    weak, strong = set(), set()

    nb_words_done = 0
    for word in def_dict:
        nb_words_done += 1
        if nb_words_done % 100 == 0:
            progress = nb_words_done / len(def_dict) * 100
            print("\r", "{:.2f}%".format(progress), end="")

        for definition_token in def_dict[word]:

            # case 0: word is used in its definition. Obvious strong pair,
            # but not interesting.
            if word == definition_token:
                continue

            # case 1: strong pair
            # Some words (like eurynome) are in vocabulary, and are used in
            # some definitions, but do not have a definition themselves. So
            # we need to be sure that definition_token is in the dictionary.
            if (definition_token in def_dict and \
               word in def_dict[definition_token]) or (definition_token in syn_dict and word in syn_dict[definition_token]):

                # use alphabetical order -> no duplicate
                w1, w2 = min(word, definition_token), max(word,definition_token)
                if not (w1,w2) in strong:
                    strong.add((w1,w2))

                # |- Artificial strong pairs generation -|
                if K > 0:

                    # to create more strong pairs, we need the embedding of
                    # definition_token. If it does not exist, can't do anything
                    if not definition_token in wordsToNum:
                        continue

                    embed_def_token = embedding[wordsToNum[definition_token]]

                    # To generate K other strong pairs, we need to find the K
                    # closest word to definition_token. Then we can create the
                    # pairs :
                    #   * (word, closest_1)
                    #   * (word, closest_2)
                    #   * ...
                    #   * (word, closest_K)
                    #
                    # Instead of taking each row of the embedding matrix and
                    # computing the cosine similarity with embed_def_token and
                    # take the K best scores, we do the dot product between the
                    # embedding matrix and embed_def_token. Because our
                    # embedding matrix is normalized, we'll get a vector
                    # containing all cosine similarities. Then we only need to
                    # find the K indexes of the maximum scores with the
                    # argpartition function. But when we compute the dot product
                    # between the matrix and the vector, we'll compute the dot
                    # product between embed_def_token and itself (hence getting
                    # a cosine sim of 1). So we need to get the K+1 best scores
                    # of similarities.

                    #start = time.time()
                    cosine_sim = embedding.dot(embed_def_token)
                    max_indexes = np.argpartition(cosine_sim, -(K+1))[-(K+1):]
                    #duree = time.time() - start
                    #print("duree:", duree, "\n")

                    for index in max_indexes:
                        close_word = numToWords[index]

                        if close_word != definition_token:
                            #print(close_word, "\t", cosine_sim[index])
                            w1, w2 = min(word,close_word), max(word,close_word)
                            strong.add((w1,w2))

            # case 2: weak pair
            else:
                w1, w2 = min(word, definition_token), max(word,definition_token)
                if not (w1,w2) in weak:
                    weak.add((w1,w2))


    # write pairs into files
    print("\n\n-- Writing pairs")
    strg_of = open("{}-K{}.txt".format(strg_fn, K), "w")
    weak_of = open("{}-K{}.txt".format(weak_fn, K), "w")

    for s in strong:
        strg_of.write(' '.join(s) + '\n')

    for s in weak:
        weak_of.write(' '.join(s) + '\n')

    strg_of.close()
    weak_of.close()

    total = (len(strong) + len(weak)) / 100.0
    print("   # strong pairs: % 8d (%.2f%%)" % (len(strong), len(strong)/total))
    print("   # weak   pairs: % 8d (%.2f%%)" % (len(weak), len(weak)/total))



def main():

    with open("data/stopwords.txt") as f:
        stopwords = f.readlines()
        stopwords = set([x.strip() for x in stopwords])

    nltk_stopwords = nltk.corpus.stopwords.words('dutch')
    stopwords.update(nltk_stopwords)
    def_dict = add_wiktionary_defs()
    syn_dict = add_syn_defs_from_json(stopwords)
    
    add_woordenlijst_pickles(syn_dict, def_dict, stopwords)

    syn_dict = {k: v for k, v in syn_dict.items() if len(v) != 0}
    def_dict = {k: v for k, v in def_dict.items() if len(v) != 0}

    generate_pairs(def_dict, syn_dict, "data/sonar-320.txt", "data/strong_pairs.txt", "data/weak_pairs.txt", 5)

if __name__ == '__main__':
    main()