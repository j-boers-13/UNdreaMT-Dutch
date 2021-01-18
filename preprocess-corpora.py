import truecase
import nltk.data
from nltk.tokenize import word_tokenize
import sys

WABLIEFT_PATH="data/wablieft-sents.txt"
DUTCHCORPUS_PATH="data/dutchcorpus-sents.txt"
TESTDUTCHCORPUS_PATH="data/dutchcorpus-sents-test.txt"
WABLIEFT_OUTPUT_PATH="data/wablieft-cleaned-sents.txt"
DUTCHCORPUS_OUTPUT_PATH="data/dutchcorpus-cleaned-sents.txt"
TESTDUTCHCORPUS_OUTPUT_PATH="data/dutchcorpus-test-cleaned.txt"

def tokenize_and_output(ifile, ofile, tokenizer):
    for line in ifile:
        for sentence in tokenizer.tokenize(line):
            tokens = word_tokenize(sentence)
            output = " ".join(tokens)
            ofile.write(output)
            ofile.write("\n")

def main():
    sent_tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')
    output_file = open(WABLIEFT_OUTPUT_PATH, 'w') 

    with open(WABLIEFT_PATH) as f:
        tokenize_and_output(f, output_file, sent_tokenizer)

    output_file.close()

    output_file = open(DUTCHCORPUS_OUTPUT_PATH, 'w')

    with open(DUTCHCORPUS_PATH) as f:
        tokenize_and_output(f, output_file, sent_tokenizer)

    output_file.close()

    output_file = open(TESTDUTCHCORPUS_OUTPUT_PATH, 'w')

    with open(TESTDUTCHCORPUS_PATH) as f:
        tokenize_and_output(f, output_file, sent_tokenizer)

    output_file.close()

if __name__ == "__main__":
    main()
