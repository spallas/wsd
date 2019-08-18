"""
Load data from SemCor files and SemEval/SensEval files.
"""

import xml.etree.ElementTree as Et
from collections import Counter, defaultdict
from typing import List, Dict

import torch
from torch import nn
from allennlp.modules.elmo import batch_to_ids
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import util


BERT_MODEL = 'bert-large-cased'

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=BERT_MODEL.endswith('-uncased'))


def str_to_token_ids(batch: List[List[str]]):
    slices_b = []
    tokens_b = []
    for seq in batch:
        bert_tok_ids = []
        slices = []
        j = 0
        seq = ['[CLS]'] + seq + ['[SEP]']
        for w in seq:
            bert_tok_ids += bert_tokenizer.encode(w)
            slices.append(slice(j, len(bert_tok_ids)))
            j = len(bert_tok_ids)
        tokens_b.append(torch.tensor(bert_tok_ids))
        slices_b.append(slices)
    tokens_b = nn.utils.rnn.pad_sequence(b_y, batch_first=True, padding_value=0)
    return tokens_b, slices_b


def str_to_char_ids(batch: List[List[str]]):
    return batch_to_ids(batch)


class SemCorDataset(Dataset):

    def __init__(self,
                 data_path='res/wsd-train/semcor+glosses_data.xml',
                 tags_path='res/wsd-train/semcor+glosses_tags.txt',
                 sense2id: Dict[str, int] = None,
                 is_training=True):
        """
        Load from XML and txt files sentences and tags.
        :param data_path: path to XML SemCor-like file.
        :param tags_path: path to text file with instance id - synset mapping
        """
        # Get sense from word instance id in data path
        with open(tags_path) as f:
            instance2senses: Dict[str, str] = {line.strip().split(' ')[0]: line.strip().split(' ')[1:] for line in f}
        senses_list = []
        for sl in instance2senses.values():
            senses_list += sl
        c = Counter(senses_list)
        # Build sense to id index if training dataset. Take from args if test dataset.
        if not sense2id:
            sense2id: Dict[str, int] = defaultdict(lambda: -1)
            # WARNING: dict value for senses not seen at training time is -1
            for i, w in enumerate(list(c.keys()), start=1):
                sense2id[w] = i
        self.sense2id = sense2id
        # 0 for monosemic words, no sense associated
        # self.senses_count = {sense2id[k]: v for k, v in c.items()}
        # if sense2id:  # i.e. loading training set
        #     self.senses_count[-1] = 0
        instance2ids: Dict[str, List[int]] = {k: list(map(lambda x: sense2id[x] if x in sense2id else -1, v))
                                              for k, v in instance2senses.items()}
        self.train_sense_map = {}
        self.docs: List[List[str]] = []
        self.senses: List[List[List[int]]] = []
        self.first_senses: List[List[int]] = []
        self.pos_tags: List[List[int]] = []
        self.vocab: Dict[str, int] = {'PAD': 0, 'UNK': 0}
        for text in tqdm(Et.parse(data_path).getroot()):
            lemmas: List[str] = []
            pos_tags: List[int] = []
            senses: List[List[int]] = []
            for sentence in text:
                for word in sentence:
                    lemma = word.attrib["lemma"]
                    lemmas.append(lemma)
                    pos_tags.append(util.pos2id[word.attrib["pos"]])
                    word_senses = instance2ids[word.attrib["id"]] if word.tag == "instance" else [0]
                    senses.append(word_senses)
                    if is_training and word.tag == "instance":
                        self.train_sense_map.setdefault(lemma, []).extend(word_senses)
                    if lemma not in self.vocab:
                        self.vocab[lemma] = len(self.vocab)
            if len(lemmas) < 10 and is_training:
                continue  # skip too short docs
            if all([x == 0 for x in [i[0] for i in senses]]):
                continue  # skip not tagged docs
            self.docs.append(lemmas)
            self.pos_tags.append(pos_tags)
            self.senses.append(senses)
            self.first_senses.append([i[0] for i in senses])
        # sort documents by length to minimize padding.
        z = zip(self.docs, self.pos_tags, self.senses, self.first_senses)
        sorted_z = sorted(z, key=lambda x: len(x[0]))  # , reverse=True)
        self.docs, self.pos_tags, self.senses, self.first_senses = map(lambda x: list(x), zip(*sorted_z))

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx], self.first_senses[idx]


class SemCorDataLoader:

    def __init__(self,
                 dataset: SemCorDataset,
                 batch_size: int,
                 win_size: int,
                 shuffle: bool = False,
                 overlap_size: int = 0,
                 return_all_senses: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.win_size = win_size
        self.do_shuffle = shuffle
        self.overlap_size = overlap_size
        self.do_return_all_senses = return_all_senses

    def __iter__(self):
        self.last_doc = 0
        self.last_offset = 0
        return self

    def __next__(self):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")


class ElmoSemCorLoader(SemCorDataLoader):
    """
    Return ELMo character ids instead of the vocab ids of the base class as input signal.
    """

    def __init__(self, dataset: SemCorDataset, batch_size: int, win_size: int, shuffle: bool = False,
                 overlap_size: int = 0, return_all_senses: bool = False):
        super().__init__(dataset, batch_size, win_size, shuffle, overlap_size, return_all_senses)

    def __next__(self):
        stop_iter = False
        b_x, b_l, b_y = [], [], []
        lengths = [len(d) for d in self.dataset.docs[self.last_doc: self.last_doc + self.batch_size]]
        end_of_docs = self.last_offset + self.win_size >= max(lengths)
        i = 0
        while len(b_x) < self.batch_size:
            if self.last_doc + i >= len(self.dataset.docs):
                stop_iter = True
                break
            n = self.last_doc + i
            m = slice(self.last_offset, self.last_offset + self.win_size)
            text_span = self.dataset.docs[n][m]
            labels = self.dataset.first_senses[n][m]
            length = len(text_span)
            # Padding
            text_span += ['.'] * (self.win_size - len(text_span))
            labels += [0] * (self.win_size - len(labels))

            i += 1
            if all([x == 0 for x in labels]):
                continue  # skip batch elem if no annotation
            b_x.append(text_span)
            b_y.append(labels)
            b_l.append(length)

        self.last_offset += self.win_size - self.overlap_size
        if end_of_docs:
            self.last_doc += self.batch_size
            self.last_offset = 0
            if stop_iter or self.last_doc >= len(self.dataset.docs):
                raise StopIteration

        return batch_to_ids(b_x), b_l, b_y


class ElmoLemmaPosLoader(SemCorDataLoader):

    def __init__(self, dataset: SemCorDataset, batch_size: int, win_size: int, shuffle: bool = False,
                 overlap_size: int = 0, return_all_senses: bool = False):
        super().__init__(dataset, batch_size, win_size, shuffle, overlap_size, return_all_senses)

    def __next__(self):
        """
        Produce one batch.
        :return: Tuple with:
            - elmo char ids: Tensor
            - lemmas: List[List[str]]
            - pos_tags: List[List[int]]
            - lengths: List[int]
            - labels: List[List[int]]
        """
        stop_iter = False
        b_x, b_l, b_p, b_y, b_z = [], [], [], [], []
        lengths = [len(d) for d in self.dataset.docs[self.last_doc: self.last_doc + self.batch_size]]
        end_of_docs = self.last_offset + self.win_size >= max(lengths)
        i = 0
        while len(b_x) < self.batch_size:
            if self.last_doc + i >= len(self.dataset.docs):
                stop_iter = True
                break
            n = self.last_doc + i
            m = slice(self.last_offset, self.last_offset + self.win_size)
            text_span = self.dataset.docs[n][m]
            labels = self.dataset.first_senses[n][m]
            pos_tags = self.dataset.pos_tags[n][m]
            all_labels = self.dataset.senses[n][m]

            length = len(text_span)
            # Padding
            text_span += ['<pad>'] * (self.win_size - length)
            labels += [0] * (self.win_size - length)
            pos_tags += [0] * (self.win_size - length)
            all_labels += [[0]] * (self.win_size + 2 - length)

            i += 1
            b_x.append(text_span)
            b_y.append(torch.tensor(labels))
            b_l.append(length)
            b_p.append(pos_tags)
            b_z.append(all_labels)

        b_y = nn.utils.rnn.pad_sequence(b_y, batch_first=True, padding_value=0)
        b_t = batch_to_ids(b_x)
        b_l = torch.tensor(b_l)

        self.last_offset += self.win_size - self.overlap_size
        if end_of_docs:
            self.last_doc += self.batch_size
            self.last_offset = 0
            if stop_iter or self.last_doc >= len(self.dataset.docs):
                raise StopIteration

        return b_t, b_x, b_p, b_l, b_y, b_z


class BertLemmaPosLoader(SemCorDataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __next__(self):
        """
        Produce one batch.
        :return: Tuple with:
            - tokens_tensor: WARNING length of SentencePiece tokenized
                             text is different from list of lemmas for example
                             use the starts vector to reconstruct original length
            - slices: List[List[Slice]]
            - lemmas: List[List[str]]
            - pos_tags: List[List[int]]
            - lengths: List[int]
            - labels: List[List[int]]
        """
        stop_iter = False
        b_t, b_x, b_l, b_p, b_y, b_s, b_z = [], [], [], [], [], [], []
        lengths = [len(d) for d in self.dataset.docs[self.last_doc: self.last_doc + self.batch_size]]
        end_of_docs = self.last_offset + self.win_size >= max(lengths)
        i = 0
        while len(b_x) < self.batch_size:
            if self.last_doc + i >= len(self.dataset.docs):
                stop_iter = True
                break
            n = self.last_doc + i
            m = slice(self.last_offset, self.last_offset + self.win_size)
            text_span = ['[CLS]'] + self.dataset.docs[n][m] + ['[SEP]']
            labels = [0] + self.dataset.first_senses[n][m] + [0]
            pos_tags = [0] + self.dataset.pos_tags[n][m] + [0]
            all_labels = [[0]] + self.dataset.senses[n][m] + [[0]]

            bert_tokens = []
            slices = []
            j = 0
            for w in text_span:
                bert_tokens += self.bert_tokenizer.encode(w)
                slices.append(slice(j, len(bert_tokens)))
                j = len(bert_tokens)
            bert_len = len(bert_tokens)
            text_len = len(text_span)
            # Padding
            text_span += ['[PAD]'] * (self.win_size + 2 - text_len)
            pos_tags += [0] * (self.win_size + 2 - text_len)
            all_labels += [[0]] * (self.win_size + 2 - text_len)

            i += 1
            # if all([x == 0 for x in labels]):
            #     continue  # skip batch elem if no annotation
            b_t.append(torch.tensor(bert_tokens))
            b_s.append(slices)
            b_x.append(text_span)
            b_l.append(bert_len)
            b_p.append(pos_tags)
            b_z.append(all_labels)
            b_y.append(torch.tensor(labels))

        b_y = nn.utils.rnn.pad_sequence(b_y, batch_first=True, padding_value=0)
        b_t = nn.utils.rnn.pad_sequence(b_t, batch_first=True, padding_value=0)
        b_l = torch.tensor(b_l)

        self.last_offset += self.win_size - self.overlap_size
        if end_of_docs:
            self.last_doc += self.batch_size
            self.last_offset = 0
            if stop_iter or self.last_doc >= len(self.dataset.docs):
                raise StopIteration

        return b_t, b_x, b_p, b_l, b_y, b_s, b_z


if __name__ == '__main__':

    data_loader = SemCorDataLoader(SemCorDataset(), batch_size=4,
                                   win_size=5, shuffle=False)

    for idx, (bx, l, by) in enumerate(data_loader):
        print(bx)
        break
