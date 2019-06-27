"""
Load data from SemCor files and SemEval/SensEval files.
"""

import xml.etree.ElementTree as Et
from typing import List, Dict

from allennlp.modules.elmo import batch_to_ids
from torch import Tensor
from torch.utils.data import Dataset

from utils import util


class SemCorDataset(Dataset):

    def __init__(self,
                 data_path='res/wsd-train/semcor_data.xml',
                 tags_path='res/wsd-train/semcor_tags.txt'):
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
        senses_list = list(set(senses_list))
        # Build sense to id index
        sense2id: Dict[str, int] = {w: i for i, w in enumerate(senses_list, 1)}  # 0 for monosemic word
        instance2ids: Dict[str, List[int]] = {k: list(map(lambda x: sense2id[x], v)) for k, v in instance2senses.items()}

        self.elmo_documents: List[Tensor] = []
        self.docs: List[List[str]] = []
        self.senses: List[List[List[int]]] = []
        self.first_senses: List[List[int]] = []
        self.pos_tags: List[List[int]] = []
        self.vocab: Dict[str, int] = {'PAD': 0, 'UNK': 0}

        for text in Et.parse(data_path).getroot():
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
                    if lemma not in self.vocab:
                        self.vocab[lemma] = len(self.vocab)
            self.elmo_documents.append(batch_to_ids([lemmas])[0])
            self.docs.append(lemmas)
            self.pos_tags.append(pos_tags)
            self.senses.append(senses)
            self.first_senses.append([i[0] for i in senses])

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
                 return_all_senses: bool = False,
                 return_pos_tags: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.win_size = win_size
        self.do_shuffle = shuffle
        self.overlap_size = overlap_size
        self.do_return_all_senses = return_all_senses
        self.do_return_pos = return_pos_tags

    def __iter__(self):
        self.last_doc = 0
        self.last_offset = 0
        return self

    def __next__(self):
        """
        Produce one batch
        :return:
            x: Tensor - shape = (batch_size x win_size)
                      - x[i][j] = token index in vocab
            lengths: Tensor
                      - shape = batch_size
                      - lengths[i] = length of text span i
            y: Tensor - shape = (batch_size x win_size)
                      - y[i][j] = sense index in vocab
        """
        b_x = []
        b_y = []
        lengths = [len(d) for d in self.dataset.docs[self.last_doc: self.last_doc + self.batch_size]]
        end_of_docs = max(lengths) <= self.last_offset + self.win_size
        for i in range(self.batch_size):
            text_span = self.dataset.docs[self.last_doc + i][self.last_offset: self.last_offset + self.win_size]
            text_span_ids = list(map(lambda x: self.dataset.vocab[x], text_span))
            labels = self.dataset.first_senses[self.last_doc + i][self.last_offset: self.last_offset + self.win_size]
            # Padding
            text_span_ids += [self.dataset.vocab['PAD']] * (self.win_size - len(text_span_ids))
            labels += [self.dataset.vocab['PAD']] * (self.win_size - len(labels))
            b_x.append(text_span_ids)
            b_y.append(labels)

        self.last_offset += self.win_size - self.overlap_size
        if end_of_docs:
            self.last_doc += self.batch_size
            self.last_offset = 0
        return b_x, lengths, b_y


if __name__ == '__main__':

    dataloader = SemCorDataLoader(SemCorDataset(), batch_size=4, win_size=5, shuffle=False)

    for idx, (bx, l, by) in enumerate(dataloader):
        print(bx)
        break
