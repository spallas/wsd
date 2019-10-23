
import io
import os
from typing import List

import fasttext
import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pytorch_transformers import BertModel, BertTokenizer
from torch import nn, optim
from tqdm import tqdm


def load_vectors(f_name):
    fin = io.open(f_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def cos_sim(a, b):
    sim = dot(a, b)/(norm(a)*norm(b))
    return sim


class BertEmbedder:

    def __init__(self):
        with torch.no_grad():
            self.model = BertModel.from_pretrained('bert-base-cased')
            self.tok = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            self.vocab = self.tok.vocab

    def embed_batch(self, sent_tok_ids, word_indices: List[int]):
        with torch.no_grad():
            last_layer, cls_emb = self.model(sent_tok_ids)
            indices = [range(len(word_indices)), word_indices]
            b_word_emb = last_layer[indices]
            return b_word_emb


class WVMapModel(nn.Module):

    def __init__(self,
                 bert_size=768,
                 ft_size=300):
        super().__init__()
        self.h_size = 1024
        self.dense = nn.Linear(ft_size, self.h_size)
        self.project = nn.Linear(self.h_size, bert_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, ft_emb, bert_emb=None):
        x = self.dense(ft_emb)
        x = self.project(x)
        if bert_emb is not None:
            loss = self.loss_fn(x, bert_emb)
            return loss, x
        else:
            return x


class MWETrainLoader:

    def __init__(self,
                 train_dataset,
                 batch_size,
                 ft_model):
        """
        :param train_dataset: path to file of format: line: int \t List[str]
        """
        self.dataset_path = train_dataset
        self.ft_model = ft_model
        self.bert_embed = BertEmbedder()
        self.bert_tok = self.bert_embed.tok
        self.data = []
        self.batch_size = batch_size
        with open(train_dataset) as f:
            for line in tqdm(f):
                l1, l2 = line.strip().split('\t')
                i, word_list = int(l1), l2.split(' ')
                self.data.append((i, word_list))
        self.offset = 0
        self.stop_flag = False

    def __iter__(self):
        self.offset = 0
        self.stop_flag = False
        return self

    def __next__(self):
        if self.stop_flag:
            raise StopIteration
        b_y, b_i, b_x = [], [], []
        with torch.no_grad():
            for i in range(self.batch_size):
                if self.offset + i >= len(self.data):
                    self.stop_flag = True
                    break
                j, sent = self.data[self.offset + i]
                ids = self.bert_tok.convert_tokens_to_ids(sent)
                b_y.append(torch.tensor(ids))
                b_i.append(j)
                b_x.append(self.ft_model[sent[j]])

            self.offset = self.offset + i

            b_x = torch.tensor(b_x)
            b_y = nn.utils.rnn.pad_sequence(b_y, True)
            b_y_i = self.bert_embed.embed_batch(b_y, b_i)
            return b_x, b_y_i


class MWEVocabExt:

    def __init__(self,
                 device,
                 saved_model_path,
                 train_text=None,
                 is_training=True,
                 ft_path='res/fasttext-vectors/cc.en.300.bin'):
        self.device = device
        self.ft_model = fasttext.load_model(ft_path)
        if is_training:
            self.device = device
            self.num_epochs = 10
            self.batch_size = 64
            self.learning_rate = 0.0001
            self.log_interval = 100
            self.checkpoint_path = saved_model_path
            self.train_text = train_text
            self.train_loader = MWETrainLoader(train_text, self.batch_size, self.ft_model)
            self.bert_embed = BertEmbedder()
            self.map_model = WVMapModel()
            self.optimizer = optim.Adam(self.map_model.parameters(), lr=self.learning_rate)
            self._maybe_load_checkpoint()
        else:
            self.checkpoint_path = saved_model_path
            self.saved_model_path = saved_model_path
            self.map_model = WVMapModel()
            self._load_best()

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch: {epoch}")
            for step, (b_x, b_y) in enumerate(self.train_loader):
                loss, _ = self.map_model(b_x, b_y)
                self._log(step, loss, epoch)
                loss.backward()
                self.optimizer.step()
            self.last_step += step

    def _load_best(self):
        if os.path.exists(self.saved_model_path):
            checkpoint = torch.load(self.saved_model_path, map_location=str(self.device))
            self.map_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Could not find any best model checkpoint: {self.saved_model_path}")

    def _log(self, step, loss, epoch_i):
        if step % self.log_interval == 0:
            print(f'\rLoss: {loss.item():.4f} ', end='')
            self._maybe_checkpoint(loss, None, epoch_i)

    def _maybe_checkpoint(self, loss, f1, epoch_i):
        current_loss = loss.item()
        if current_loss < self.min_loss:
            min_loss = current_loss
            torch.save({
                'epoch': epoch_i,
                'last_step': self.last_step,
                'model_state_dict': self.map_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'min_loss': min_loss
            }, self.checkpoint_path)

    def _maybe_load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.map_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            self.last_step = checkpoint['last_step']
            self.min_loss = checkpoint['min_loss']
            print(f"Loaded checkpoint from: {self.checkpoint_path}")
            if self.last_epoch >= self.num_epochs:
                print("Training finished for this checkpoint")
        else:
            self.last_epoch = 0
            self.last_step = 0
            self.min_loss = 1e3

    def get_mwe_embedding_matrix(self, vocab: List[str]):
        """
        Get pseudo-BERT vector for each string in vocab.
        Transform vectors with current model and return.
        :param vocab:
        :return:
        """
        mwe_ft_vectors = []
        for w in vocab:
            w = w.replace('_', ' ')
            vec = self.ft_model.get_sentence_vector(w)
            mwe_ft_vectors.append(torch.tensor(vec))
        mwe_ft_matrix = torch.stack(mwe_ft_vectors, dim=0)
        bert_vectors = []
        # batch_size = 128
        for i in range(0, mwe_ft_matrix.shape[0]):  # , batch_size):
            bert_vec = self.map_model(mwe_ft_matrix[i].unsqueeze(0))
            bert_vectors.append(bert_vec[0])
        mwe_bert_matrix = torch.stack(bert_vectors)
        return mwe_bert_matrix


if __name__ == '__main__':

    # model = fasttext.load_model('../res/fasttext-vectors/cc.en.300.bin')
    #
    # v = model.get_word_vector('door')
    # vvv = model.get_sentence_vector('room access')
    # v4 = model.get_sentence_vector('door')
    # vv = model.get_word_vector('room access')
    #
    # print(cos_sim(v, vv))
    # print(cos_sim(vv, vvv))
    # print(cos_sim(v, vvv))
    # print(cos_sim(vvv, v4))
    # print(cos_sim(v, v4))

    mwe_model = MWEVocabExt('cpu', 'saved_weights/ft2bert.pth', is_training=False)
    a = mwe_model.map_model(torch.tensor(mwe_model.ft_model.get_sentence_vector('financial support'))).detach()
    aa = mwe_model.map_model(torch.tensor(mwe_model.ft_model.get_sentence_vector('financial backing'))).detach()
    b = mwe_model.map_model(torch.tensor(mwe_model.ft_model.get_sentence_vector('financial obligation'))).detach()

    print(cos_sim(a, aa))
    print(cos_sim(a, b))
