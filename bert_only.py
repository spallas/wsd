import os
import xml.etree.ElementTree as Et
from collections import defaultdict
from typing import Dict, List

import torch
from pytorch_transformers import BertTokenizer, BertConfig, BertModel
from torch import optim, nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm import tqdm

from data_preprocessing import BERT_MODEL
from train import BaseTrainer
from utils import util
from utils.config import BertWsdConfig
from utils.util import NOT_AMB_SYMBOL, UNK_SENSE
from wsd import BaselineWSD


def build_sense2id(tags_path='res/wsd-train/semcor+glosses_tags.txt',
                   dict_path='res/dictionaries/senses.txt'):
    sense2id: Dict[str, int] = defaultdict(lambda: NOT_AMB_SYMBOL)
    with open(tags_path) as f:
        senses_set = set()
        for line in f:
            senses_set.update(line.strip().split(' ')[1:])
    with open(dict_path, 'w') as f:
        for i, w in enumerate(senses_set, start=1):
            sense2id[w] = i
            print(f"{w} {i}", file=f)
    return sense2id


def load_sense2id(dict_path='res/dictionaries/senses.txt'):
    with open(dict_path) as f:
        sense2id = {line.strip().split(' ')[0]: int(line.strip().split(' ')[1]) for line in f}
    return sense2id


class FlatSemCorDataset(Dataset):

    def __init__(self,
                 data_path='res/wsd-train/semcor+glosses_data.xml',
                 tags_path='res/wsd-train/semcor+glosses_tags.txt',
                 sense_dict='res/dictionaries/senses.txt'):
        with open(tags_path) as f:
            instance2senses: Dict[str, str] = {line.strip().split(' ')[0]: line.strip().split(' ')[1:] for line in f}
        sense2id = load_sense2id(sense_dict) if os.path.exists(sense_dict) else build_sense2id(tags_path, sense_dict)
        instance2ids: Dict[str, List[int]] = {k: list(map(lambda x: sense2id[x] if x in sense2id else UNK_SENSE, v))
                                              for k, v in instance2senses.items()}
        self.num_tags = len(sense2id)
        self.train_sense_map = {}
        self.dataset_lemmas = []
        self.first_senses = []
        self.all_senses = []
        self.pos_tags = []
        for text in tqdm(Et.parse(data_path).getroot()):
            for sentence in text:
                for word in sentence:
                    self.dataset_lemmas.append(word.attrib['lemma'])
                    self.pos_tags.append(util.pos2id[word.attrib['pos']])
                    word_senses = instance2ids[word.attrib['id']] if word.tag == 'instance' else [NOT_AMB_SYMBOL]
                    self.all_senses.append(word_senses)
                    self.first_senses.append(word_senses[0])
                    self.train_sense_map.setdefault(word.attrib['lemma'], []).extend(word_senses)

    def __len__(self):
        return len(self.dataset_lemmas)

    def __getitem__(self, idx):
        return {'lemma': self.dataset_lemmas[idx],
                'pos': self.pos_tags[idx],
                'sense': self.first_senses[idx],
                'all_senses': self.all_senses[idx]}


class BertSimpleLoader:

    def __init__(self,
                 dataset: FlatSemCorDataset,
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
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=False)

    def __iter__(self):
        self.last_offset = 0
        self.stop_flag = False
        return self

    def __next__(self):
        if self.stop_flag:
            raise StopIteration
        b_t, b_x, b_l, b_p, b_y, b_s, b_z = [], [], [], [], [], [], []
        for i in range(self.batch_size):
            n = self.last_offset + i
            m = n + self.win_size
            if m > len(self.dataset):
                self.stop_flag = True
                m = len(self.dataset)
            text_window = ['[CLS]'] + self.dataset.dataset_lemmas[n:m] + ['[SEP]']
            text_window += ['[PAD]'] * (self.win_size + 2 - len(text_window))
            pos_tags = [0] + self.dataset.pos_tags[n:m] + [0]
            pos_tags += [0] * (self.win_size + 2 - len(pos_tags))
            sense_labels = [NOT_AMB_SYMBOL] + self.dataset.first_senses[n:m] + [NOT_AMB_SYMBOL]
            sense_labels += [NOT_AMB_SYMBOL] * (self.win_size + 2 - len(sense_labels))
            all_senses = [[NOT_AMB_SYMBOL]] + self.dataset.all_senses[n:m] + [[NOT_AMB_SYMBOL]]
            all_senses += [[NOT_AMB_SYMBOL]] * (self.win_size + 2 - len(all_senses))

            bert_ids, slices = [], []
            j = 0
            for w in text_window:
                bert_ids += self.bert_tokenizer.encode(w)
                slices.append(slice(j, len(bert_ids)))
                j = len(bert_ids)
            bert_len = len(bert_ids)

            b_t.append(torch.tensor(bert_ids))
            b_x.append(text_window)
            b_l.append(bert_len)
            b_p.append(torch.tensor(pos_tags))
            b_y.append(torch.tensor(sense_labels))
            b_s.append(slices)
            b_z.append(all_senses)
            if self.stop_flag:
                break
        self.last_offset = m
        b_y = nn.utils.rnn.pad_sequence(b_y, batch_first=True, padding_value=NOT_AMB_SYMBOL)
        b_t = nn.utils.rnn.pad_sequence(b_t, batch_first=True, padding_value=0)
        b_l = torch.tensor(b_l)
        return b_t, b_x, b_p, b_l, b_y, b_s, b_z


class BertWSD(BaselineWSD):

    def __init__(self, device, num_senses, max_len, encoder_embed_dim, d_model):
        super().__init__(num_senses, max_len)
        self.device = device
        self.encoder_embed_dim = encoder_embed_dim
        self.d_model = d_model
        self.bert_config = BertConfig.from_pretrained(BERT_MODEL)
        self.bert_model = BertModel(self.bert_config)

        self.dense_1 = nn.Linear(self.encoder_embed_dim, self.d_model)
        self.dense_2 = nn.Linear(self.d_model, self.tagset_size)
        self.ce_loss = CrossEntropyLoss()

    def forward(self, token_ids, lengths, slices, labels=None):
        max_len = token_ids.shape[1]
        bert_mask = torch.arange(max_len)\
                         .expand(len(lengths), max_len)\
                         .to(self.device) < lengths.unsqueeze(1)
        x = self.bert_model(token_ids, attention_mask=bert_mask)[0]
        batch_x = []
        for i in range(x.shape[0]):
            s = x[i]
            m = [torch.mean(s[sl, :], dim=-2) for sl in slices[i]]
            mt = torch.cat(m).reshape(-1, self.encoder_embed_dim)
            batch_x.append(mt)
        x = torch.cat(batch_x).reshape(len(batch_x), -1, self.encoder_embed_dim)
        x = self.dense_1(x)
        logits = self.dense_2(x)
        outputs = logits
        if labels is not None:
            active_loss = labels.view(-1) != NOT_AMB_SYMBOL
            if not active_loss.any():
                loss = torch.tensor(0)
            else:
                active_logits = logits.view(-1, self.tagset_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.ce_loss(active_logits, active_labels)
            outputs = (logits, loss)
        return outputs


class BertWsdTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        self.encoder_embed_dim = kwargs['encoder_embed_dim']
        self.d_model = kwargs['d_model']
        self.learning_rate = kwargs['learning_rate']
        super().__init__(**kwargs)

    def _setup_training(self, train_data, train_tags, eval_data, eval_tags):
        # Load data
        dataset = FlatSemCorDataset(train_data, train_tags)
        self.sense2id = load_sense2id()
        self.train_sense_map = dataset.train_sense_map
        num_tags = dataset.num_tags + 1
        self.data_loader = BertSimpleLoader(dataset, batch_size=self.batch_size,
                                            win_size=self.window_size)
        self.tokenizer = self.data_loader.bert_tokenizer
        eval_dataset = FlatSemCorDataset(data_path=eval_data, tags_path=eval_tags)
        self.eval_loader = BertSimpleLoader(eval_dataset, batch_size=self.batch_size,
                                            win_size=self.window_size, overlap_size=8)
        # Build model
        self.model = BertWSD(self.device, num_tags, self.window_size,
                             self.encoder_embed_dim, self.d_model)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._maybe_load_checkpoint()

    def _setup_testing(self, train_data, train_tags, test_data, test_tags):
        dataset = FlatSemCorDataset(train_data, train_tags)
        self.train_sense_map = dataset.train_sense_map
        num_tags = dataset.num_tags + 1
        test_dataset = FlatSemCorDataset(data_path=test_data, tags_path=test_tags)
        self.test_loader = BertSimpleLoader(test_dataset, batch_size=self.batch_size,
                                            win_size=self.window_size, overlap_size=8)
        self.model = BertWSD(self.device, num_tags, self.window_size,
                             self.encoder_embed_dim, self.d_model)
        self._load_best()
        self.model.eval()
        self.model.to(self.device)

    def train_epoch(self, epoch_i):
        for step, (b_t, b_x, b_p, b_l, b_y, b_s, b_z) in enumerate(self.data_loader, self.last_step):
            self.model.zero_grad()
            scores, loss = self.model(b_t.to(self.device),
                                      b_l.to(self.device),
                                      b_s,
                                      b_y.to(self.device))
            loss.backward()
            if step % self.log_interval == 0:
                print(f'\rLoss: {loss.item():.4f} ', end='')
                self._plot('Train loss', loss.item(), step)
                self._gpu_mem_info()
                f1 = self._evaluate(epoch_i)
                self._maybe_checkpoint(loss, f1, epoch_i)
                self._plot('Dev F1', f1, step)
                self.model.train()  # return to train mode after evaluation
            self.optimizer.step()
        self.last_step += step

    def train(self):
        self.model.train()
        for epoch in range(self.last_epoch + 1, self.num_epochs + 1):
            print(f'\nEpoch: {epoch}')
            self.train_epoch(epoch)

    def test(self, loader):
        """
        Evaluate on all test dataset.
        """
        if not loader:
            loader = self.test_loader
        with torch.no_grad():
            pred, true, z = [], [], []
            for step, (b_t, b_x, b_p, b_l, b_y, b_s, b_z) in enumerate(loader):
                scores = self.model(b_t.to(self.device),
                                    b_l.to(self.device),
                                    b_s)
                pred += [item for seq in self._select_senses(scores, b_t, b_x, b_p, None, b_y) for item in seq]
                true += [item for seq in b_y.tolist() for item in seq]
                z += [item for seq in b_z for item in seq]

            true_eval, pred_eval = [], []
            for i in range(len(true)):
                if true[i] == NOT_AMB_SYMBOL:
                    continue
                else:
                    if pred[i] in z[i]:
                        true_eval.append(pred[i])
                    else:
                        true_eval.append(true[i])
                    pred_eval.append(pred[i])
            f1 = self._print_metrics(true_eval, pred_eval)
        return f1


if __name__ == '__main__':
    c: BertWsdConfig = BertWsdConfig.from_json_file('conf/bert_wsd_conf.json')
    t = BertWsdTrainer(**c.__dict__)
    t.train()
