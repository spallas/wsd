
from nltk.corpus import wordnet as wn
from nltk.tokenize import treebank, mwe


class WNTokenizer:

    WN_VOCAB = wn.all_lemma_names()

    def __init__(self):

        self._base_tok = treebank.TreebankWordTokenizer()
        self.wn_mwe_list = [tuple(w.split('_')) for w in self.WN_VOCAB if '_' in w]
        self._mwe_tok = mwe.MWETokenizer(self.wn_mwe_list)

    def tokenize(self, text: str):
        base_tokenized = self._base_tok.tokenize(text)
        tokenized = self._mwe_tok.tokenize(base_tokenized)
        return tokenized



