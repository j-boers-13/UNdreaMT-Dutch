import truecase
from nltk.tokenize import word_tokenize
import sys

WABLIEFT_PATH="data/wablieft-sents.txt"
OUTPUT_PATH="data/wablieft-sents-cleaned.txt"

def main():
    output_file = open(OUTPUT_PATH, 'a') 

    # print command line arguments
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            for line in f:
                tokens = word_tokenize(line)
                output = " ".join(tokens)
                output_file.write(output)
                output_file.write("\n")
    else:
        with open(WABLIEFT_PATH) as f:
            for line in f:
                tokens = word_tokenize(line)
                output = " ".join(tokens)
                output_file.write(output)
                output_file.write("\n")
    output_file.close()


if __name__ == "__main__":
    main()
