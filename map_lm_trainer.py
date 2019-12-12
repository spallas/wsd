import copy
import logging
from typing import Set, Dict, List

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM
from scipy.special import softmax
from torch import nn

from ft2bert.ft2bert import MWEVocabExt
from train import BaseTrainer
from utils import util
from utils.util import NOT_AMB_SYMBOL


def mask_subsets(masks: List[int]) -> List[List[int]]:
    num_subsets = 5
    mask_subs = [[]] * num_subsets
    for i, m in enumerate(masks):
        mask_subs[i % num_subsets].append(m)
    return mask_subs


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
        set_list = [self._get_lemmas(S) for S in self.synsets]
        common_lemmas = set_list[0].intersection(*set_list[1:])
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


class BertForMLMExt(BertForMaskedLM):

    def extend_vocab(self, extended_weights):
        self.cls.predictions.decoder.weight = nn.Parameter(torch.cat([self.cls.predictions.decoder.weight,
                                                                      extended_weights]),
                                                           False)
        self.cls.predictions.bias = nn.Parameter(torch.cat([self.cls.predictions.bias,
                                                            torch.zeros([extended_weights.size(0)])]),
                                                 False)


class MaskedLMWrapper:

    def __init__(self, device, mwe_vocab: List[str]):
        self.device = device
        self.mwe_vocab = mwe_vocab
        with torch.no_grad():
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            self.bert_vocab_size = len(self.bert_tokenizer.vocab)
            self.language_model: BertForMLMExt = BertForMLMExt.from_pretrained('bert-base-cased',
                                                                               output_hidden_states=True)
            mwe_model = MWEVocabExt(self.device, 'saved_weights/ft2bert.pth', is_training=False)
            self.mwe_embed_matrix = mwe_model.get_mwe_embedding_matrix(self.mwe_vocab)
            self.language_model.extend_vocab(self.mwe_embed_matrix)
            self.language_model.eval()

    def encode_lemma(self, lemma) -> List[int]:
        if lemma not in self.mwe_vocab:
            return self.bert_tokenizer.encode(lemma)
        else:
            return [self.mwe_vocab.index(lemma) + self.bert_vocab_size]

    def sub_tokenize_and_map(self, batch: List[List[str]]):
        batch_out, word_map = [], []
        for sent in batch:
            t_sent, t_map = [], []
            for w in sent:
                t_map.append(len(t_sent))
                t_sent.extend(self.encode_lemma(w))
            batch_out.append(t_sent)
            word_map.append(t_map)
        return batch_out, word_map

    def predict_masks(self,
                      masked_sent_list: List[List[str]],
                      masks: List[List[int]]) -> List[torch.Tensor]:
        """
        Add also lemmas out of LM vocabulary here
        """
        b_size = len(masked_sent_list)
        encoded_sent_list, mask_map = self.sub_tokenize_and_map(masked_sent_list)
        outputs_ = self.language_model(torch.tensor(encoded_sent_list))
        outputs = outputs_[0]
        # hidden_states_list = outputs_[1]
        # print(len(hidden_states_list))
        # for hs in hidden_states_list:
        #     print(hs.shape)
        mapped_masks = []
        for i, m in enumerate(masks):
            mm = []
            for j in m:
                mm.append(mask_map[i][j])
            mapped_masks.append(mm)
        prediction_scores = [outputs[i, j] for i in range(b_size) for j in mapped_masks[i]]
        return prediction_scores

    def best_match(self, lemmas: List[int]):
        """
        Manage also lemmas out of LM vocabulary here
        """
        pass


class TrainerLM(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.masked_lm = MaskedLMWrapper(self.device, mwe_vocab=self._load_mwe_vocab())
        self.lsw_cache = {}
        self.distinct_lemmas: Dict[str, LemmaSensesWords] = {}
        self.batches_scores = []
        self.K = 10
        logging.debug(f'Using top {self.K}')

    @staticmethod
    def _load_mwe_vocab():
        mwe_word_list = []
        with open('res/dictionaries/mwe_err.txt') as f:
            for line in f:
                mwe_word_list.append(line.strip())
        return mwe_word_list

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

            text = ['[CLS]'] + sent + ['[SEP]']
            masked_indices = [k + 1 for k, _ in enumerate(text[1:-1]) if b_labels[i][k] != NOT_AMB_SYMBOL]
            mask_subs = mask_subsets(masked_indices)
            text_batch = []
            for masks in mask_subs:
                text_ = copy.deepcopy(text)
                for m in masks:
                    text_[m] = '[MASK]'
                text_batch.append(text_)

            predictions = self.masked_lm.predict_masks(text_batch, mask_subs)

            for m, prediction in zip(masked_indices, predictions):
                k = m - 1
                w = text[m]
                pos = util.id2wnpos[b_pos[i][k]]

                lm_ids, lm_scores, net_score = [], [], {}
                ext_v_scores = []
                unique_lsw = self.get_lemma_senses_words(w, pos)
                for S in unique_lsw.get_filtered_by_name(self.sense2id):
                    s_id = self.sense2id[S.name()]
                    syn_tok_ids = [tid for l in unique_lsw.lemmas(S)
                                   for tid in self.masked_lm.encode_lemma(l)]
                    prediction_subset = prediction[syn_tok_ids, ]
                    top_k = torch.topk(prediction[syn_tok_ids, ], k=min(self.K, prediction_subset.size(0)))
                    top_indices = top_k[1].tolist()
                    for top_i in top_indices:
                        if top_i >= 28996:
                            logging.info(f"Used extended vocab with index: {top_i}")
                    top_k = top_k[0].tolist()
                    ext_tok_ids = [tid for tid in syn_tok_ids if tid >= 28996]
                    if not ext_tok_ids:
                        ext_v_scores.append(0)
                    else:
                        # print(f"{len(ext_tok_ids)} ", end='')
                        top_k = torch.topk(prediction[ext_tok_ids, ], k=min(5, len(ext_tok_ids)))[0].tolist()
                        ext_v_scores.append(sum(top_k))
                    lm_ids.append(s_id)
                    lm_scores.append(sum(top_k))
                    net_score[s_id] = b_scores[i, k, s_id]
                if not lm_scores:
                    continue
                combined_scores = [s * 0.1 + se * 0.0 for s, se in zip(softmax(lm_scores), softmax(ext_v_scores))]
                lm_score = dict(zip(lm_ids, softmax(combined_scores)))  # else just lm_scores
                for s_id in lm_score:
                    if w in self.train_sense_map:
                        b_scores[i, k, s_id] = (net_score[s_id] * lm_score[s_id]) ** 0.5
                    else:
                        b_scores[i, k, s_id] = lm_score[s_id]
        return np.argmax(b_scores, -1).tolist()

    def _build_model(self):
        pass
