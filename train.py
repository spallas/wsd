import os
import pprint
import warnings
from typing import Set

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from pytorch_transformers import BertForMaskedLM, BertTokenizer
from scipy.special import softmax
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, f1_score
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from data_preprocessing import SemCorDataset, ElmoSemCorLoader, \
    ElmoLemmaPosLoader, BertLemmaPosLoader
from utils import util
from utils.config import TransformerConfig
from wsd import SimpleWSD, BertTransformerWSD

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.manual_seed(42)
np.random.seed(42)


class BaseTrainer:

    def __init__(self,
                 num_epochs=40,
                 batch_size=32,
                 window_size=64,
                 checkpoint_path='saved_weights/baseline_elmo_checkpoint.pt',
                 log_interval=400,
                 train_data='res/wsd-train/semcor+glosses_data.xml',
                 train_tags='res/wsd-train/semcor+glosses_tags.txt',
                 eval_data='res/wsd-test/se07/se07.xml',
                 eval_tags='res/wsd-test/se07/se07.txt',
                 test_data='res/wsd-train/test_data.xml',
                 test_tags='res/wsd-train/test_tags.txt',
                 report_path='logs/baseline_elmo_report.txt',
                 is_training=True,
                 **kwargs):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval
        self._plot_server = None
        self.report_path = report_path
        self.model = None
        self.optimizer = None
        self.min_loss = np.inf
        self.data_loader = None
        self.eval_loader = None
        self.sense2id = None
        self.train_sense_map = {}
        self.last_step = 0

        self.best_model_path = self.checkpoint_path + '.best'

        # Using single GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_training:
            self._setup_training(train_data, train_tags, eval_data, eval_tags)
        else:
            self._setup_testing(train_data, train_tags, test_data, test_tags)

    def _setup_training(self, train_data, train_tags, eval_data, eval_tags):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def _setup_testing(self, train_data, train_tags, test_data, test_tags):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def train_epoch(self, epoch_i):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def train(self):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def test(self, loader):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def _evaluate(self, num_epoch):
        print("\nEvaluating...", flush=True)
        self.model.eval()
        f1 = self.test(self.eval_loader)
        self._save_best(f1, num_epoch)
        return f1

    def _select_senses(self, b_scores, b_vec, b_str, b_pos, b_lengths, b_labels):
        """
        Get the max of scores only of possible senses for a given lemma+pos
        :param b_scores: shape = (batch_s x win_s x sense_vocab_s)
        :param b_vec: unused
        :param b_str:
        :param b_pos:
        :param b_lengths:
        :return:
        """
        def to_ids(synsets):
            return set([self.sense2id[x.name()] for x in synsets])

        def set2padded(s: Set[int]):
            arr = np.array(list(s))
            return np.pad(arr, (0, b_scores.shape[-1] - len(s)), 'edge')

        b_impossible_senses = []
        # we will set to 0 senses not in WordNet for given lemma.
        for i, sent in enumerate(b_str):
            impossible_senses = []
            for j, lemma in enumerate(sent):
                sense_ids = to_ids(wn.synsets(lemma, pos=util.id2wnpos[b_pos[i][j]]))
                if lemma in self.train_sense_map:
                    sense_ids &= set(self.train_sense_map[lemma])
                padded = set2padded(set(range(b_scores.shape[-1])) - sense_ids)
                impossible_senses.append(padded)
            b_impossible_senses.append(impossible_senses)
        b_scores = b_scores.cpu().numpy()
        b_impossible_senses = np.array(b_impossible_senses)
        np.put_along_axis(b_scores, b_impossible_senses, -np.inf, axis=-1)
        return np.argmax(b_scores, -1).tolist()

    def _plot(self, name, value, step):
        if not self._plot_server:
            self._plot_server = SummaryWriter(log_dir='logs')
        self._plot_server.add_scalar(name, value, step)

    def _print_metrics(self, true_eval, pred_eval):
        with open(self.report_path, 'w') as fo:
            print(classification_report(
                true_eval,
                pred_eval,
                digits=3),
                file=fo)
        f1 = f1_score(true_eval, pred_eval, average='micro')
        print(f"F1 = {f1}")
        return f1

    def _maybe_checkpoint(self, loss, f1, epoch_i):
        current_loss = loss.item()
        if current_loss < self.min_loss:
            min_loss = current_loss
            torch.save({
                'epoch': epoch_i,
                'last_step': self.last_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_loss': current_loss,
                'min_loss': min_loss,
                'f1': self.best_f1_micro
            }, self.checkpoint_path)

    def _maybe_load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            self.last_step = checkpoint['last_step']
            self.min_loss = checkpoint['min_loss']
            self.best_f1_micro = checkpoint['f1']
            print(f"Loaded checkpoint from: {self.checkpoint_path}")
            if self.last_epoch >= self.num_epochs:
                print("Training finished for this checkpoint")
        else:
            self.last_epoch = 0
            self.last_step = 0
            self.min_loss = 1e3
            self.best_f1_micro = 0.0

    def _load_best(self):
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=str(self.device))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Could not find any best model checkpoint: {self.best_model_path}")

    def _save_best(self, f1, epoch_i):
        if f1 > self.best_f1_micro:
            self.best_f1_micro = f1
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': self.model.state_dict(),
                'f1': self.best_f1_micro
            }, self.best_model_path)

    @staticmethod
    def _gpu_mem_info():
        if torch.cuda.is_available():  # check if memory is leaking
            print(f'Allocated GPU memory: '
                  f'{torch.cuda.memory_allocated() / 1_000_000} MB', end='')


class TrainerLM(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.language_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.language_model.eval()
        self.all_syn_lemmas = {}

    def _select_senses(self, b_scores, b_vec, b_str, b_pos, b_lengths, b_labels):
        """
        Use Language model to get a second score and use geometric mean on scores.
        :param b_scores: shape = (batch_s x win_s x sense_vocab_s)
        :param b_vec:
        :param b_str:
        :param b_pos:
        :param b_lengths:
        :return:
        """
        def to_ids(synsets):
            return [self.sense2id[x.name()] for x in synsets]

        def set2padded(s: Set[int]):
            arr = np.array(list(s))
            return np.pad(arr, (0, b_scores.shape[-1] - len(s)), 'edge')

        def get_lemmas(synset):
            lemmas = synset.lemma_names()
            for s in synset.hyponyms():
                lemmas += s.lemma_names()
            for s in synset.hypernyms():
                lemmas += s.lemma_names()
            return lemmas

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
                if b_labels[i][k] != 0:  # i.e. sense tagged word
                    # text = ['[CLS]'] + sent + ['[SEP]']
                    # text[k+1] = '[MASK]'
                    text = [] + sent
                    text[k] = '[MASK]'
                    tokenized_text = []
                    for ww in text:
                        tokenized_text += self.bert_tokenizer.tokenize(ww)
                    masked_index = tokenized_text.index('[MASK]')
                    indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    outputs = self.language_model(tokens_tensor)
                    predictions = outputs[0]
                    probabilities = torch.nn.Softmax(dim=0)(predictions[0, masked_index])

                    lm_ids, lm_scores = [], []
                    net_score = {}
                    for S in wn.synsets(w, pos=util.id2wnpos[b_pos[i][k]]):
                        s_id = self.sense2id[S.name()]
                        if S not in self.all_syn_lemmas:
                            self.all_syn_lemmas[S] = get_lemmas(S)
                        syn_tok_ids = []
                        for lemma in self.all_syn_lemmas[S]:
                            tokenized = self.bert_tokenizer.tokenize(lemma)
                            tok_ids = self.bert_tokenizer.convert_tokens_to_ids(tokenized)
                            syn_tok_ids += tok_ids
                        top_k = torch.topk(probabilities[syn_tok_ids, ], k=10)[0].tolist() \
                            if len(syn_tok_ids) > 5 else probabilities[syn_tok_ids, ].tolist()
                        s_score = sum(top_k)
                        lm_ids.append(s_id)
                        lm_scores.append(s_score)
                        net_score[s_id] = b_scores[i, k, s_id]
                    lm_score = {k: v for k, v in zip(lm_ids, softmax(lm_scores))}
                    for s_id in lm_score:
                        if w in self.train_sense_map:
                            b_scores[i, k, s_id] = (net_score[s_id] * lm_score[s_id]) ** 0.5
                        else:
                            b_scores[i, k, s_id] = lm_score[s_id]

        return np.argmax(b_scores, -1).tolist()

    def _setup_training(self, train_data, train_tags, eval_data, eval_tags):
        pass

    def _setup_testing(self, train_data, train_tags, test_data, test_tags):
        pass

    def train_epoch(self, epoch_i):
        pass

    def train(self):
        pass

    def test(self, loader):
        pass


class ElmoTrainerLM(TrainerLM):

    def __init__(self,
                 hidden_size=1024,
                 num_layers=2,
                 learning_rate=0.001,
                 elmo_weights='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',
                 elmo_options='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json',
                 elmo_size=128,
                 **kwargs):
        self.learning_rate = learning_rate
        self.elmo_weights = elmo_weights
        self.elmo_options = elmo_options
        self.elmo_size = elmo_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def _setup_training(self, train_data, train_tags, eval_data, eval_tags):
        # Load data
        dataset = SemCorDataset(train_data, train_tags)
        self.sense2id = dataset.sense2id
        self.train_sense_map = dataset.train_sense_map
        num_tags = len(self.sense2id) + 1
        self.data_loader = ElmoLemmaPosLoader(dataset, batch_size=self.batch_size,
                                              win_size=self.window_size)
        eval_dataset = SemCorDataset(data_path=eval_data,
                                     tags_path=eval_tags,
                                     sense2id=self.sense2id,
                                     is_training=False)
        self.eval_loader = ElmoLemmaPosLoader(eval_dataset, batch_size=self.batch_size,
                                              win_size=self.window_size, overlap_size=8)
        # Build model
        self.model = SimpleWSD(num_tags,
                               self.window_size,
                               self.elmo_weights, self.elmo_options,
                               self.elmo_size, self.hidden_size, self.num_layers)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._maybe_load_checkpoint()

    def _setup_testing(self, train_data, train_tags, test_data, test_tags):
        # Load data
        dataset = SemCorDataset(train_data, train_tags)
        self.sense2id = dataset.sense2id
        self.train_sense_map = dataset.train_sense_map
        num_tags = len(self.sense2id) + 1
        dataset = SemCorDataset(data_path=test_data,
                                tags_path=test_tags,
                                sense2id=self.sense2id,
                                is_training=False)
        self.test_loader = ElmoLemmaPosLoader(dataset, batch_size=self.batch_size,
                                              win_size=self.window_size)
        # Build model
        self.model = SimpleWSD(num_tags,
                               self.window_size,
                               self.elmo_weights, self.elmo_options,
                               self.elmo_size, self.hidden_size, self.num_layers)
        self._load_best()
        self.model.eval()
        self.model.to(self.device)

    def train_epoch(self, epoch_i):
        for step, (b_t, b_x, b_p, b_l, b_y, b_z) in enumerate(self.data_loader, self.last_step):
            self.model.zero_grad()
            self.model.h, self.model.cell = map(lambda x: x.to(self.device),
                                                self.model.init_hidden(len(b_y)))
            scores = self.model(b_t.to(self.device), b_l.to(self.device))
            loss = self.model.loss(scores, b_y.to(self.device))
            loss.backward()

            if step % self.log_interval == 0:
                print(f'\rLoss: {loss.item():.4f} ', end='')
                self._plot('Train loss', loss.item(), step)
                self._gpu_mem_info()
                f1 = self._evaluate(epoch_i)
                self._maybe_checkpoint(loss, f1, epoch_i)
                self._plot('Dev F1', f1, step)
                self.model.train()  # return to train mode after evaluation

            clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            self.optimizer.step()  # update the weights
        self.last_step += step

    def train(self):
        self.model.train()
        for epoch in range(self.last_epoch + 1, self.num_epochs + 1):
            print(f'\nEpoch: {epoch}')
            self.train_epoch(epoch)

    def test(self, loader=None):
        """
        Evaluate on test dataset.
        """
        if not loader:
            loader = self.test_loader
        with torch.no_grad():
            pred, true, z = [], [], []
            for step, (b_t, b_x, b_p, b_l, b_y, b_z) in enumerate(loader):
                self.model.zero_grad()
                self.model.h, self.model.cell = map(lambda x: x.to(self.device),
                                                    self.model.init_hidden(len(b_y)))
                scores = self.model(b_t.to(self.device), b_l.to(self.device))
                pred += self._select_senses(scores, b_t, b_x, b_p, b_l, b_y)
                true += b_y.tolist()
                z += b_z
            true_flat, pred_flat, z_flat = [item for sublist in true for item in sublist], \
                                           [item for sublist in pred for item in sublist], \
                                           [item for sublist in z for item in sublist]
            true_eval, pred_eval = [], []
            for i in range(len(true_flat)):
                if true_flat[i] == 0:
                    continue
                else:
                    if pred_flat[i] in z_flat[i]:
                        true_eval.append(pred_flat[i])
                    else:
                        true_eval.append(true_flat[i])
                    pred_eval.append(pred_flat[i])
            f1 = self._print_metrics(true_eval, pred_eval)
        return f1


class TransformerTrainer(TrainerLM):

    def __init__(self, config: TransformerConfig, **kwargs):
        self.config = config
        super().__init__(**kwargs)

    def _setup_training(self, train_data, train_tags, eval_data, eval_tags):
        # Load data
        dataset = SemCorDataset(train_data, train_tags)
        self.sense2id = dataset.sense2id
        self.train_sense_map = dataset.train_sense_map
        num_tags = len(self.sense2id) + 1
        self.data_loader = BertLemmaPosLoader(dataset, batch_size=self.batch_size,
                                              win_size=self.window_size)
        self.tokenizer = self.data_loader.bert_tokenizer
        eval_dataset = SemCorDataset(data_path=eval_data,
                                     tags_path=eval_tags,
                                     sense2id=self.sense2id,
                                     is_training=False)
        self.eval_loader = BertLemmaPosLoader(eval_dataset, batch_size=self.batch_size,
                                              win_size=self.window_size, overlap_size=8)
        # Build model
        self.model = BertTransformerWSD(self.device, num_tags, self.window_size, self.config)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self._maybe_load_checkpoint()

    def _setup_testing(self, train_data, train_tags, test_data, test_tags):
        dataset = SemCorDataset(train_data, train_tags)
        self.sense2id = dataset.sense2id
        self.train_sense_map = dataset.train_sense_map
        num_tags = len(self.sense2id) + 1
        test_dataset = SemCorDataset(data_path=test_data,
                                     tags_path=test_tags,
                                     sense2id=self.sense2id,
                                     is_training=False)
        self.test_loader = BertLemmaPosLoader(test_dataset, batch_size=self.batch_size,
                                              win_size=self.window_size, overlap_size=8)
        self.model = BertTransformerWSD(self.device, num_tags, self.window_size, self.config)
        self._load_best()
        self.model.eval()
        self.model.to(self.device)

    def train_epoch(self, epoch_i):
        for step, (b_t, b_x, b_p, b_l, b_y, b_s, b_z) in enumerate(self.data_loader, self.last_step):
            self.model.zero_grad()
            b_lengths = torch.tensor([sum([1 for w in sent if w != '[PAD]']) for sent in b_x]).to(self.device)
            b_pos = torch.tensor([p_row[:b_lengths.max().item()] for p_row in b_p]).to(self.device)
            scores = self.model(b_t.to(self.device),
                                b_l.to(self.device),
                                b_s,
                                b_lengths,
                                b_pos)
            loss = self.model.loss(scores, b_y.to(self.device))
            # provide starts to aggregate scores of sub-words
            loss.backward()

            if step % self.log_interval == 0:
                print(f'\rLoss: {loss.item():.4f} ', end='')
                self._plot('Train loss', loss.item(), step)
                self._gpu_mem_info()
                f1 = self._evaluate(epoch_i)
                self._maybe_checkpoint(loss, f1, epoch_i)
                self._plot('Dev F1', f1, step)
                self.model.train()  # return to train mode after evaluation

            clip_grad_norm_(parameters=self.model.parameters(), max_norm=5.0)
            self.optimizer.step()  # update the weights
        self.last_step += step

    def train(self):
        self.model.train()
        for epoch in range(self.last_epoch + 1, self.num_epochs + 1):
            print(f'\nEpoch: {epoch}')
            self.train_epoch(epoch)

    def test(self, loader=None):
        """
        Evaluate on all test dataset.
        """
        if not loader:
            loader = self.test_loader
        with torch.no_grad():
            pred, true, z = [], [], []
            for step, (b_t, b_x, b_p, b_l, b_y, b_s, b_z) in enumerate(loader):
                b_lengths = torch.tensor([sum([1 for w in sent if w != '[PAD]']) for sent in b_x]).to(self.device)
                b_pos = torch.tensor([p_row[:b_lengths.max().item()] for p_row in b_p]).to(self.device)
                scores = self.model(b_t.to(self.device),
                                    b_l.to(self.device),
                                    b_s,
                                    b_lengths,
                                    b_pos)
                pred += self._select_senses(scores, b_t, b_x, b_p, b_l, b_y)
                true += b_y.tolist()
                z += b_z
            true_flat, pred_flat, z_flat = [item for sublist in true for item in sublist], \
                                           [item for sublist in pred for item in sublist], \
                                           [item for sublist in z for item in sublist]
            true_eval, pred_eval = [], []
            for i in range(len(true_flat)):
                if true_flat[i] == 0:
                    continue
                else:
                    if pred_flat[i] in z_flat[i]:
                        true_eval.append(pred_flat[i])
                    else:
                        true_eval.append(true_flat[i])
                    pred_eval.append(pred_flat[i])
            f1 = self._print_metrics(true_eval, pred_eval)
        return f1


class WSDNetTrainer(BaseTrainer):
    def _setup_training(self, train_data, train_tags, eval_data, eval_tags):
        pass

    def _setup_testing(self, train_data, train_tags, test_data, test_tags):
        pass

    def train_epoch(self, epoch_i):
        pass

    def train(self):
        pass

    def test(self, loader):
        pass


if __name__ == '__main__':
    c = TransformerConfig.from_json_file('conf/transformer_wsd_conf.json')
    t = TransformerTrainer(c, **c.__dict__)
    t.train()
