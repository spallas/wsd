import re
from collections import Counter

from typing import List

from tqdm import tqdm

from data_preprocessing import FlatSemCorDataset
from ft2bert import BertEmbedder
from mweLM.wn_tokenizer import WNTokenizer


def main():

    # output: one sentence per line
    # one train word per line
    output_file = 'data/test_bert_examples.txt'
    num_sent_per_word = 10

    semcor_lemmas = FlatSemCorDataset().dataset_lemmas

    emb = BertEmbedder()
    words = [w for w in list(emb.vocab.keys()) if re.match(r'[a-zA-Z]', w[0])]
    loaded_counter = Counter(words)
    tok = emb.tok

    sentences = []

    while(len(semcor_lemmas)) > 0:
        sentences.append(' '.join(semcor_lemmas[:50]))
        semcor_lemmas = semcor_lemmas[min(len(semcor_lemmas), 50):]

    with open(output_file, 'w') as fo:
        for line in tqdm(sentences):
            tokenized: List = tok.tokenize(line.strip())
            for w in words:
                if loaded_counter[w] < num_sent_per_word + 1:
                    if w in tokenized:
                        example = tokenized
                        example_i = tokenized.index(w)
                        loaded_counter[w] += 1
                        print(f"{example_i}\t{' '.join(example)}", file=fo)


if __name__ == '__main__':
    main()
