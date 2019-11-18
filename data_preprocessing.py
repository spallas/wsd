"""
Load data from SemCor files and SemEval/SensEval files.
"""
import logging
import os
import random
import xml.etree.ElementTree as Et
from collections import Counter, defaultdict
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

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


class FlatLoader:

    def __init__(self,
                 dataset: FlatSemCorDataset,
                 batch_size: int,
                 win_size: int,
                 pad_symbol: str,
                 overlap: int = 0,
                 randomize: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.win_size = win_size
        self.pad_symbol = pad_symbol
        self.randomize = randomize
        self.overlap = overlap if not randomize else 0

    def __iter__(self):
        self.offset = 0
        self.index_list = list(range((len(self.dataset) // self.batch_size)+1, self.batch_size))
        if self.randomize:
            random.shuffle(self.index_list)
        return self

    def __next__(self):
        b_t, b_x, b_l, b_p, b_y, b_s, b_z = [], [], [], [], [], [], []
        try:
            self.offset = self.index_list.pop()
        except IndexError:
            raise StopIteration
        for i in range(self.batch_size):
            n = max(self.offset + (i * self.win_size), 0)
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
        b_y = nn.utils.rnn.pad_sequence(b_y, batch_first=True, padding_value=NOT_AMB_SYMBOL)
        return b_x, b_p, b_y, b_z


class CachedEmbedLoader:

    HALF = 0
    SINGLE = 1

    def __init__(self,
                 device,
                 cache_file: str,
                 model_path: str,
                 batch_mul: int = 1,
                 batch_size: int = 1,
                 flat_loader: FlatLoader = None):
        self.flat_loader = None
        self.embed = None
        self.npz_file = None
        self.cache_file = cache_file
        self.offset = 0
        self.cache = []
        self.dataset = None
        self.device = device
        self.batch_mul = batch_mul
        self.stop_flag = False
        self.second_half = None
        self.batch_size = batch_size
        self.stop_flag = False
        self.index_list = None
        if os.path.exists(self.cache_file):
            logging.info(f'Loading cache from {self.cache_file}')
            self._load_cache()
        else:
            self.flat_loader = flat_loader
            self.embed = RobertaAlignedEmbed(device, model_path)
            self._create_cache()

    def _create_cache(self):
        for i, (b_x, b_p, b_y, b_z) in tqdm(enumerate(self.flat_loader)):
            self.cache.append(self.embed(b_x).cpu().numpy())
        np.savez(self.cache_file, *self.cache)

    def _load_cache(self):
        self.npz_file = np.load(self.cache_file)

    def __iter__(self):
        self.offset = 0
        self.second_half = None
        return self

    def __next__(self):
        if self.batch_mul == self.HALF:
            if self.second_half is None:
                try:
                    self.offset = self.index_list.pop()
                except IndexError:
                    raise StopIteration
                batch = self.npz_file[f'arr_{self.offset}'] if len(self.cache) == 0 else self.cache[self.offset]
                if len(batch) < self.batch_size:
                    return torch.tensor(batch).to(self.device)
                batch_ = batch[:self.batch_size]
                self.second_half = batch[self.batch_size:]
                return torch.tensor(batch_).to(self.device)
            else:
                second_half = self.second_half
                self.second_half = None
                if len(second_half) > 0:
                    return torch.tensor(second_half).to(self.device)
                else:
                    raise StopIteration
        elif self.batch_mul > self.SINGLE:
            try:
                self.offset = self.index_list.pop()
            except IndexError:
                raise StopIteration
            batch_a = self.npz_file[f'arr_{self.offset}'] if len(self.cache) == 0 else self.cache[self.offset]
            batches = [batch_a]
            for i in range(self.batch_mul - 1):
                try:
                    self.offset += 1
                    batch_b = self.npz_file[f'arr_{self.offset}'] if len(self.cache) == 0 else self.cache[self.offset]
                    batches.append(batch_b)
                except (KeyError, IndexError):
                    break
            batch = np.stack(batches)
            return torch.tensor(batch).to(self.device)
        else:
            try:
                self.offset = self.index_list.pop()
            except IndexError:
                raise StopIteration
            batch = self.npz_file[f'arr_{self.offset}'] if len(self.cache) == 0 else self.cache[self.offset]
            self.offset += 1
            return torch.tensor(batch).to(self.device)


if __name__ == '__main__':

    dataset_ = FlatSemCorDataset()
    data_loader = FlatLoader(dataset_, 100, 100, 'PAD')

    for bx in enumerate(data_loader):
        print(bx)
        break
