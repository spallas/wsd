"""
Load data from SemCor files and SemEval/SensEval files.
"""
import logging
import os
import xml.etree.ElementTree as Et
from collections import Counter, defaultdict
from typing import List, Dict

import torch
from transformers import BertTokenizer
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from models import RobertaAlignedEmbed
from utils import util
from utils.util import UNK_SENSE, NOT_AMB_SYMBOL, is_ascii


def build_sense2id(dict_path='res/dictionaries/senses.txt',
                   tags_path='res/wsd-train/semcor_tags.txt',
                   test_tags_path='res/wsd-train/test_tags.txt'):
    sense2id: Dict[str, int] = defaultdict(lambda: NOT_AMB_SYMBOL)
    with open(tags_path) as f, open(test_tags_path) as ff:
        senses_set = set()
        for line in f:
            senses_set.update(line.strip().split(' ')[1:])
        for line in ff:
            senses_set.update(line.strip().split(' ')[1:])
    senses_list = list(sorted(senses_set))
    with open(dict_path, 'w') as f:
        for i, w in enumerate(senses_list, start=1):
            sense2id[w] = i
            print(f"{w} {i}", file=f)
    return sense2id


def load_sense2id(dict_path='res/dictionaries/senses.txt',
                  tags_path='res/wsd-train/semcor_tags.txt',
                  test_tags_path='res/wsd-train/test_tags.txt'):
    if os.path.exists(dict_path):
        with open(dict_path) as f:
            sense2id = {line.strip().split(' ')[0]: int(line.strip().split(' ')[1]) for line in f}
    else:
        sense2id = build_sense2id(dict_path, tags_path, test_tags_path)
    return sense2id


class FlatSemCorDataset(Dataset):

    def __init__(self,
                 data_path='res/wsd-train/semcor_data.xml',
                 tags_path='res/wsd-train/semcor_tags.txt',
                 sense_dict='res/dictionaries/senses.txt'):
        with open(tags_path) as f:
            instance2senses: Dict[str, str] = {line.strip().split(' ')[0]: line.strip().split(' ')[1:] for line in f}
        sense2id = load_sense2id(sense_dict, tags_path=tags_path)
        instance2ids: Dict[str, List[int]] = {k: list(map(lambda x: sense2id[x] if x in sense2id else UNK_SENSE, v))
                                              for k, v in instance2senses.items()}
        self.num_tags = len(sense2id)
        self.train_sense_map = {}
        self.dataset_lemmas = []
        self.first_senses = []
        self.all_senses = []
        self.pos_tags = []
        for text in tqdm(Et.parse(data_path).getroot(), desc=f'Loading data from {data_path}'):
            for sentence in text:
                for word in sentence:
                    lemma = word.attrib['lemma'] if is_ascii(word.attrib['lemma']) else '#'
                    self.dataset_lemmas.append(lemma)
                    self.pos_tags.append(util.pos2id[word.attrib['pos']])
                    word_senses = instance2ids[word.attrib['id']] if word.tag == 'instance' else [NOT_AMB_SYMBOL]
                    self.all_senses.append(word_senses)
                    self.first_senses.append(word_senses[0])
                    self.train_sense_map.setdefault(lemma, Counter()).update(word_senses)
        logging.info(f'Loaded dataset from {data_path}/{tags_path}')
        logging.info(f'Sense dict in {sense_dict}')

    def __len__(self):
        return len(self.dataset_lemmas)

    def __getitem__(self, idx):
        return {'lemma': self.dataset_lemmas[idx],
                'pos': self.pos_tags[idx],
                'sense': self.first_senses[idx],
                'all_senses': self.all_senses[idx]}


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


class BertLemmaPosLoader(SemCorDataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

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


class FlatLoader:

    def __init__(self,
                 dataset: FlatSemCorDataset,
                 batch_size: int,
                 win_size: int,
                 pad_symbol: str,
                 do_overlap: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.win_size = win_size
        self.pad_symbol = pad_symbol
        self.do_overlap = do_overlap

    def __iter__(self):
        self.last_offset = 0
        self.stop_flag = False
        return self

    def __next__(self):
        if self.stop_flag:
            raise StopIteration
        b_t, b_x, b_l, b_p, b_y, b_s, b_z = [], [], [], [], [], [], []
        for i in range(self.batch_size):
            overlap = 16 if self.do_overlap else 0
            n = max(self.last_offset + (i * self.win_size) - overlap, 0)
            m = n + self.win_size
            if m > len(self.dataset):
                self.stop_flag = True
                m = len(self.dataset)
            text_window = self.dataset.dataset_lemmas[n:m]
            text_window += [self.pad_symbol] * (self.win_size - len(text_window))
            pos_tags = self.dataset.pos_tags[n:m]
            pos_tags += [0] * (self.win_size - len(pos_tags))
            sense_labels = self.dataset.first_senses[n:m]
            sense_labels += [NOT_AMB_SYMBOL] * (self.win_size - len(sense_labels))
            all_senses = self.dataset.all_senses[n:m]
            all_senses += [[NOT_AMB_SYMBOL]] * (self.win_size - len(all_senses))

            b_x.append(text_window)
            b_p.append(pos_tags)
            b_y.append(torch.tensor(sense_labels))
            b_z.append(all_senses)
            if self.stop_flag:
                break
        self.last_offset = m
        b_y = nn.utils.rnn.pad_sequence(b_y, batch_first=True, padding_value=NOT_AMB_SYMBOL)
        return b_x, b_p, b_y, b_z


class CachedEmbedLoader:

    def __init__(self,
                 cache_file: str,
                 device,
                 model_path: str,
                 flat_loader: FlatLoader = None):
        self.flat_loader = None
        self.embed = None
        self.npz_file = None
        self.cache_file = cache_file
        self.offset = 0
        self.cache = []
        if os.path.exists(cache_file):
            self.npz_file = np.load(self.cache_file)
        else:
            self.flat_loader = flat_loader
            self.embed = RobertaAlignedEmbed(device, model_path)
            self._create_cache()

    def _create_cache(self):
        for i, (b_x, b_p, b_y, b_z) in enumerate(self.flat_loader):
            self.cache.append(self.embed(b_x).numpy())
        np.savez(self.cache_file, *self.cache)

    def __iter__(self):
        self.offset = 0
        return self

    def __next__(self):
        try:
            batch = self.npz_file[f'arr_{self.offset}'] if len(self.cache) == 0 else self.cache[self.offset]
            self.offset += 1
            return batch
        except (KeyError, IndexError):
            raise StopIteration


if __name__ == '__main__':

    dataset_ = FlatSemCorDataset()
    data_loader = FlatLoader(dataset_, 100, 100, 'PAD')

    for bx in enumerate(data_loader):
        print(bx)
        break
