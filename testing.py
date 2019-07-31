import os

import torch
from allennlp.modules.elmo import batch_to_ids
from torch import optim

from data_preprocessing import SemCorDataLoader, SemCorDataset, ElmoSemCorLoader
from train import BaseTrainer
from wsd import SimpleWSD


class ZerosDataLoader(SemCorDataLoader):

    def __init__(self, dataset: SemCorDataset, batch_size: int, win_size: int, shuffle: bool = False,
                 overlap_size: int = 0, return_all_senses: bool = False, return_pos_tags: bool = False):
        super().__init__(dataset, batch_size, win_size, shuffle, overlap_size, return_all_senses, return_pos_tags)

    def __next__(self):
        stop_iter = False
        b_x, b_l, b_y = [], [], []
        lengths = [len(d) for d in self.dataset.docs[self.last_doc: self.last_doc + self.batch_size]]
        end_of_docs = self.last_offset + self.win_size >= max(lengths)
        for i in range(self.batch_size):
            if self.last_doc + i >= len(self.dataset.docs):
                stop_iter = True
                break
            text_span = self.dataset.docs[self.last_doc + i][self.last_offset: self.last_offset + self.win_size]
            labels = self.dataset.first_senses[self.last_doc + i][self.last_offset: self.last_offset + self.win_size]
            length = len(text_span)
            # Padding
            text_span = ['-'] * self.win_size
            labels += [0] * (self.win_size - len(labels))
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


class OneBatchDataLoader(SemCorDataLoader):

    def __init__(self, dataset: SemCorDataset, batch_size: int, win_size: int, shuffle: bool = False,
                 overlap_size: int = 0, return_all_senses: bool = False, return_pos_tags: bool = False):
        super().__init__(dataset, batch_size, win_size, shuffle, overlap_size, return_all_senses, return_pos_tags)
        self.one_batch = None

    def __next__(self):
        stop_iter = False
        b_x, b_l, b_y = [], [], []
        lengths = [len(d) for d in self.dataset.docs[self.last_doc: self.last_doc + self.batch_size]]
        end_of_docs = self.last_offset + self.win_size >= max(lengths)
        for i in range(self.batch_size):
            if self.last_doc + i >= len(self.dataset.docs):
                stop_iter = True
                break
            text_span = self.dataset.docs[self.last_doc + i][self.last_offset: self.last_offset + self.win_size]
            labels = self.dataset.first_senses[self.last_doc + i][self.last_offset: self.last_offset + self.win_size]
            length = len(text_span)
            # Padding
            text_span += ['PAD'] * (self.win_size - len(text_span))
            labels += [0] * (self.win_size - len(labels))
            b_x.append(text_span)
            b_y.append(labels)
            b_l.append(length)

        self.last_offset += self.win_size - self.overlap_size
        if end_of_docs:
            self.last_doc += self.batch_size
            self.last_offset = 0
            if stop_iter or self.last_doc >= len(self.dataset.docs):
                raise StopIteration
        if self.one_batch is None:
            self.one_batch = (batch_to_ids(b_x), b_l, b_y)
        return self.one_batch


class ZerosBaseTrainer(BaseTrainer):

    def __init__(self,
                 learning_rate=0.001,
                 num_epochs=8,
                 batch_size=32,
                 checkpoint_path='saved_weights/zeros/checkpoint.pt'):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        # Using single GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load data
        dataset = SemCorDataset()
        self.data_loader = ZerosDataLoader(dataset, batch_size=batch_size, win_size=32)
        eval_dataset = SemCorDataset(data_path='res/wsd-test/se07/se07.xml',
                                     tags_path='res/wsd-test/se07/se07.txt')
        self.eval_loader = ElmoSemCorLoader(eval_dataset, batch_size=batch_size,
                                            win_size=32, overlap_size=8)
        # Build model
        self.model = SimpleWSD(self.data_loader)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_f1_micro = 0.0

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            self.min_loss = checkpoint['min_loss']
            print(f"Loaded checkpoint from: {checkpoint_path}")
            if last_epoch >= num_epochs:
                print("Training finished for this checkpoint")
        else:
            self.last_epoch = 0
            self.min_loss = 1e3

    def _evaluate(self,
                  num_epoch,
                  eval_report='logs/zeros_report.txt',
                  best_model_path='saved_weights/zeros/best_checkpoint.pt'):
        super()._evaluate(num_epoch)


class OneBatchBaseTrainer(BaseTrainer):

    def __init__(self,
                 learning_rate=0.001,
                 num_epochs=8,
                 batch_size=32,
                 checkpoint_path='saved_weights/one_batch/checkpoint.pt'):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        # Using single GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load data
        dataset = SemCorDataset()
        self.data_loader = OneBatchDataLoader(dataset, batch_size=batch_size, win_size=32)
        eval_dataset = SemCorDataset(data_path='res/wsd-test/se07/se07.xml',
                                     tags_path='res/wsd-test/se07/se07.txt')
        self.eval_loader = ElmoSemCorLoader(eval_dataset, batch_size=batch_size,
                                            win_size=32, overlap_size=8)
        # Build model
        self.model = SimpleWSD(self.data_loader)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_f1_micro = 0.0

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            self.min_loss = checkpoint['min_loss']
            print(f"Loaded checkpoint from: {checkpoint_path}")
            if last_epoch >= num_epochs:
                print("Training finished for this checkpoint")
        else:
            self.last_epoch = 0
            self.min_loss = 1e3
