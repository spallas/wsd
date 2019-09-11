from functools import reduce
from typing import Set, Dict

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from pytorch_transformers import BertTokenizer, BertForMaskedLM
from scipy.special import softmax

from train import BaseTrainer
from utils import util
from utils.util import NOT_AMB_SYMBOL


class LemmaSensesWords:
    """
    Lists for each sense of a lemma-pos term the words for each sense
    that are unique to the sense.
    """

    synset_lemmas_cache = {}

    def __init__(self, lemma, pos):
        self.lemma = lemma
        self.pos = pos
        self.synsets = wn.synsets(lemma, pos=pos)
        common_lemmas = reduce(lambda s, ss: s | ss, [self._get_lemmas(S) for S in self.synsets])
        self.synsets_unique_lemmas = {}
        for s in self.synsets:
            s_lemmas = self._get_lemmas(s) - common_lemmas
            self.synsets_unique_lemmas[s] = s_lemmas

    def lemmas(self, synset):
        return self.synsets_unique_lemmas[synset]

    def _get_lemmas(self, synset):
        if synset in self.synset_lemmas_cache:
            return self.synset_lemmas_cache[synset]
        lemmas = synset.lemma_names()
        for s in synset.hyponyms():
            lemmas += s.lemma_names()
        for s in synset.hypernyms():
            lemmas += s.lemma_names()
        lemmas = set(lemmas)
        self.synset_lemmas_cache[synset] = lemmas
        return lemmas

    def get_filtered_by_name(self, filter_names):
        return [s for s in self.synsets if s.name() in filter_names]


class TrainerLM(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.language_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.language_model.eval()
        self.lsw_cache = {}
        self.distinct_lemmas: Dict[str, LemmaSensesWords] = {}
        self.batches_scores = []

    def get_lemma_senses_words(self, lemma, pos):
        key = lemma + '|' + pos
        if key in self.lsw_cache:
            return self.lsw_cache[key]
        else:
            lsw = LemmaSensesWords(lemma, pos)
            self.lsw_cache[key] = lsw
            return lsw

    def _select_senses(self, b_scores, b_str, b_pos, b_labels):
        """
        Use Language model to get a second score and use geometric mean on scores.
        :param b_scores: shape = (batch_s x win_s x sense_vocab_s)
        :param b_str:
        :param b_pos:
        :return:
        """
        def to_ids(synsets):
            return set([self.sense2id.get(x.name(), 0) for x in synsets]) - {0}

        def set2padded(s: Set[int]):
            arr = np.array(list(s))
            return np.pad(arr, (0, b_scores.shape[-1] - len(s)), 'edge')

        b_impossible_senses = []
        # we will set to 0 senses not in WordNet for given lemma.
        for i, sent in enumerate(b_str):
            impossible_senses = []
            for j, lemma in enumerate(sent):
                sense_ids = to_ids(wn.synsets(lemma, pos=util.id2wnpos[b_pos[i][j]]))
                padded = set2padded(set(range(b_scores.shape[-1])) - set(sense_ids))
                impossible_senses.append(padded)
            b_impossible_senses.append(impossible_senses)
        b_scores = b_scores.cpu().numpy()
        b_impossible_senses = np.array(b_impossible_senses)
        np.put_along_axis(b_scores, b_impossible_senses, np.min(b_scores), axis=-1)
        b_scores = softmax(b_scores, -1)

        # Update b_scores with geometric mean with language model score.
        for i, sent in enumerate(b_str):
            for k, w in enumerate(sent):
                if b_labels[i][k] != NOT_AMB_SYMBOL:  # i.e. sense tagged word
                    pos = util.id2wnpos[b_pos[i][k]]
                    if len(wn.synsets(w, pos=pos)) == 1:
                        continue
                    k_bert = k + 1
                    text = ['[CLS]'] + sent + ['[SEP]']
                    text[k_bert] = '[MASK]'
                    tokens_tensor = torch.tensor([self.bert_tokenizer.encode(' '.join(text))])
                    masked_index = (tokens_tensor.squeeze() == 103).nonzero().squeeze().item()  # [MASK] index is 103
                    # Run LM
                    outputs = self.language_model(tokens_tensor)
                    predictions = outputs[0][0, masked_index]
                    predictions = torch.nn.Softmax(dim=0)(predictions)

                    lm_ids, lm_scores, net_score = [], [], {}
                    unique_lsw = self.get_lemma_senses_words(w, pos)
                    for S in unique_lsw.get_filtered_by_name(self.sense2id):
                        s_id = self.sense2id[S.name()]
                        syn_tok_ids = [tid for l in unique_lsw.lemmas(S) for tid in self.bert_tokenizer.encode(l)]
                        if len(syn_tok_ids) > 10:
                            top_k = torch.topk(predictions[syn_tok_ids, ], k=10)[0].tolist()
                        else:
                            top_k = predictions[syn_tok_ids, ].tolist()
                        s_score = sum(top_k)
                        lm_ids.append(s_id)
                        lm_scores.append(s_score)
                        net_score[s_id] = b_scores[i, k, s_id]
                    if not lm_scores:
                        continue
                    lm_score = dict(zip(lm_ids, softmax(lm_scores)))
                    for s_id in lm_score:
                        if w in self.train_sense_map:
                            b_scores[i, k, s_id] = (net_score[s_id] * lm_score[s_id]) ** 0.5
                        else:
                            b_scores[i, k, s_id] = lm_score[s_id]

        return np.argmax(b_scores, -1).tolist()

    def _build_model(self):
        pass
