import argparse
import os
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

from data_preprocessing import FlatSemCorDataset, \
    load_sense2id, FlatLoader
from utils import util
from utils.config import RobertaTransformerConfig
from utils.util import NOT_AMB_SYMBOL
from wsd import ElmoTransformerWSD, RobertaTransformerWSD, BertTransformerWSD, BaselineWSD

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.manual_seed(42)
np.random.seed(42)


class BaseTrainer:

    def __init__(self,
                 num_epochs=40,
                 batch_size=32,
                 window_size=64,
                 learning_rate=0.0001,
                 checkpoint_path='saved_weights/baseline_elmo_checkpoint.pt',
                 log_interval=400,
                 train_data='res/wsd-train/semcor+glosses_data.xml',
                 train_tags='res/wsd-train/semcor+glosses_tags.txt',
                 eval_data='res/wsd-test/se07/se07.xml',
                 eval_tags='res/wsd-test/se07/se07.txt',
                 test_data='res/wsd-train/test_data.xml',
                 test_tags='res/wsd-train/test_tags.txt',
                 sense_dict='res/dictionaries/senses.txt',
                 report_path='logs/baseline_elmo_report.txt',
                 pad_symbol='PAD',
                 is_training=True,
                 **kwargs):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval
        self._plot_server = None
        self.report_path = report_path
        self.model = None
        self.optimizer = None
        self.min_loss = np.inf
        self.data_loader = None
        self.eval_loader = None
        self.test_loader = None
        self.train_sense_map = {}
        self.last_step = 0

        self.best_model_path = self.checkpoint_path + '.best'
        self.sense2id = load_sense2id(sense_dict, train_tags, test_tags)
        self.pad_symbol = pad_symbol

        dataset = FlatSemCorDataset(train_data, train_tags)
        self.train_sense_map = dataset.train_sense_map
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._build_model()

        if is_training:
            self.data_loader = FlatLoader(dataset, batch_size=self.batch_size, win_size=self.window_size,
                                          pad_symbol=self.pad_symbol)
            self._setup_training(train_data, train_tags, eval_data, eval_tags)
        else:
            self._setup_testing(train_data, train_tags, test_data, test_tags)

    def _build_model(self):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def _setup_training(self, train_data, train_tags, eval_data, eval_tags):
        eval_dataset = FlatSemCorDataset(data_path=eval_data, tags_path=eval_tags)
        self.eval_loader = FlatLoader(eval_dataset, batch_size=self.batch_size, win_size=self.window_size,
                                      pad_symbol=self.pad_symbol)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._maybe_load_checkpoint()

    def _setup_testing(self, train_data, train_tags, test_data, test_tags):
        test_dataset = FlatSemCorDataset(data_path=test_data, tags_path=test_tags)
        self.test_loader = FlatLoader(test_dataset, batch_size=self.batch_size, win_size=self.window_size,
                                      pad_symbol=self.pad_symbol)
        self._load_best()
        self.model.eval()
        self.model.to(self.device)

    def train_epoch(self, epoch_i):
        step = 0
        for step, (b_x, b_p, b_y, b_z) in enumerate(self.data_loader, self.last_step):
            self.model.zero_grad()
            scores = self.model(b_x)
            loss = self.model.loss(scores, b_y.to(self.device))
            loss.backward()
            self._log(step, loss, epoch_i)
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
        """
        if not loader:
            loader = self.test_loader
        with torch.no_grad():
            pred, true, z = [], [], []
            for step, (b_x, b_p, b_y, b_z) in enumerate(loader):
                scores = self.model(b_x)
                true += [item for seq in b_y.tolist() for item in seq]
                pred += [item for seq in self._select_senses(scores, b_x, b_p, b_y) for item in seq]
                z += [item for seq in b_z for item in seq]
            return self._get_metrics(true, pred, z)

    def _evaluate(self, num_epoch):
        print("\nEvaluating...", flush=True)
        self.model.eval()
        f1 = self.test(self.eval_loader)
        self._save_best(f1, num_epoch)
        return f1

    def _select_senses(self, b_scores, b_str, b_pos, b_labels):
        """
        Get the max of scores only of possible senses for a given lemma+pos
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
                if b_labels[i, j] == NOT_AMB_SYMBOL:
                    sense_ids = set()
                else:
                    sense_ids = to_ids(wn.synsets(lemma, pos=util.id2wnpos[b_pos[i][j]]))
                # if lemma in self.train_sense_map:
                #     sense_ids &= set(self.train_sense_map[lemma])
                padded = set2padded(set(range(b_scores.shape[-1])) - sense_ids)
                impossible_senses.append(padded)
            b_impossible_senses.append(impossible_senses)
        b_scores = b_scores.cpu().numpy()
        b_impossible_senses = np.array(b_impossible_senses)
        np.put_along_axis(b_scores, b_impossible_senses, np.min(b_scores), axis=-1)
        return np.argmax(b_scores, -1).tolist()

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

    def _get_metrics(self, true, pred, alternatives=None):
        true_eval, pred_eval = [], []
        for i in range(len(true)):
            if true[i] == NOT_AMB_SYMBOL:
                continue
            else:
                if alternatives is None or pred[i] in alternatives[i]:
                    true_eval.append(pred[i])
                else:
                    true_eval.append(true[i])
                pred_eval.append(pred[i])
        return self._print_metrics(true_eval, pred_eval)

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

    def _log(self, step, loss, epoch_i):
        if step % self.log_interval == 0:
            print(f'\rLoss: {loss.item():.4f} ', end='')
            self._plot('Train loss', loss.item(), step)
            self._gpu_mem_info()
            f1 = self._evaluate(epoch_i)
            self._maybe_checkpoint(loss, f1, epoch_i)
            self._plot('Dev F1', f1, step)
            self.model.train()  # return to train mode after evaluation

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

    def _plot(self, name, value, step):
        if not self._plot_server:
            self._plot_server = SummaryWriter(log_dir='logs')
        self._plot_server.add_scalar(name, value, step)

    @staticmethod
    def _gpu_mem_info():
        if torch.cuda.is_available():  # check if memory is leaking
            print(f'Allocated GPU memory: '
                  f'{torch.cuda.memory_allocated() / 1_000_000} MB', end='')


class TrainerLM(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.language_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.language_model.eval()
        self.all_syn_lemmas = {}

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
                if b_labels[i][k] != NOT_AMB_SYMBOL:  # i.e. sense tagged word
                    if len(wn.synsets(w, pos=util.id2wnpos[b_pos[i][k]])) == 1:
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
                    possible_synsets = [s for s in wn.synsets(w, pos=util.id2wnpos[b_pos[i][k]]) if s.name() in self.sense2id]
                    for S in possible_synsets:
                        s_id = self.sense2id[S.name()]
                        if S not in self.all_syn_lemmas:
                            self.all_syn_lemmas[S] = get_lemmas(S)
                        syn_tok_ids = [tok_id for l in self.all_syn_lemmas[S]
                                                  for tok_id in self.bert_tokenizer.encode(l)]
                        top_k = torch.topk(predictions[syn_tok_ids, ], k=10)[0].tolist() \
                            if len(syn_tok_ids) > 10 else predictions[syn_tok_ids, ].tolist()
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


class ElmoLSTMTrainer(BaseTrainer):

    def _build_model(self):
        self.model = BaselineWSD(self.device, len(self.sense2id) + 1, self.window_size,
                                 self.elmo_weights, self.elmo_options, self.elmo_size,
                                 self.hidden_size, self.num_layers)

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


class ElmoTransformerTrainer(BaseTrainer):

    def __init__(self,
                 num_layers=2,
                 elmo_weights='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',
                 elmo_options='res/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json',
                 elmo_size=128,
                 d_model=512,
                 num_heads=4,
                 **kwargs):
        self.elmo_weights = elmo_weights
        self.elmo_options = elmo_options
        self.elmo_size = elmo_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        super().__init__(**kwargs)

    def _build_model(self):
        self.model = ElmoTransformerWSD(self.device, len(self.sense2id) + 1, self.window_size, self.elmo_weights,
                                        self.elmo_options, self.elmo_size, self.d_model,
                                        self.num_heads, self.num_layers)


class RobertaDenseTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_model(self):
        pass


class RobertaTrainer(BaseTrainer):

    def __init__(self,
                 num_layers=2,
                 d_embeddings=1024,
                 d_model=2048,
                 num_heads=4,
                 model_path='res/roberta.large',
                 **kwargs):
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_embeddings = d_embeddings
        self.num_heads = num_heads
        self.model_path = model_path
        super().__init__(**kwargs)

    def _build_model(self):
        self.model = RobertaTransformerWSD(self.device, len(self.sense2id) + 1, self.window_size,
                                           self.model_path, self.d_embeddings, self.d_model,
                                           self.num_heads, self.num_layers)


class BertTransformerTrainer(BaseTrainer):

    def __init__(self,
                 d_model=512,
                 num_layers=2,
                 num_heads=4,
                 bert_model='bert-large-cased',
                 **kwargs):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.bert_model = bert_model
        super().__init__(**kwargs)

    def _build_model(self):
        self.model = BertTransformerWSD(self.device, len(self.sense2id) + 1, self.window_size,
                                        self.d_model, self.num_heads, self.num_layers,
                                        self.bert_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train with different models")
    parser.add_argument("-m", "--model", type=str, help="model name", required=True)
    parser.add_argument("-c", "--config", type=str, help="config JSON file", required=True)

    c = RobertaTransformerConfig.from_json_file('conf/roberta_tr_conf_2.json')
    t = RobertaTrainer(**c.__dict__)
    t.train()
