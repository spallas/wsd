import os
import warnings
from typing import Set

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer
from scipy.special import softmax
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, f1_score
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from data_preprocessing import SemCorDataset, ElmoSemCorLoader, ElmoLemmaPosLoader
from utils import util
from wsd import SimpleWSD

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.manual_seed(42)
np.random.seed(42)


class BaseTrainer:

    def __init__(self,
                 learning_rate=0.001,
                 num_epochs=40,
                 batch_size=64,
                 checkpoint_path='saved_weights/baseline_elmo/checkpoint.pt',
                 is_training=False):

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self._plot_server = None

        # Using single GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_training:
            self._setup_training()
        else:
            self._setup_testing()

    def _setup_training(self):
        # Load data
        dataset = SemCorDataset()
        self.data_loader = ElmoSemCorLoader(dataset, batch_size=self.batch_size, win_size=32)
        eval_dataset = SemCorDataset(data_path='res/wsd-test/se07/se07.xml',
                                     tags_path='res/wsd-test/se07/se07.txt',
                                     sense2id=dataset.sense2id)
        self.eval_loader = ElmoLemmaPosLoader(eval_dataset, batch_size=self.batch_size,
                                              win_size=32, overlap_size=8)
        # Build model
        self.model = SimpleWSD(self.data_loader)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_f1_micro = 0.0

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            self.min_loss = checkpoint['min_loss']
            print(f"Loaded checkpoint from: {self.checkpoint_path}")
            if self.last_epoch >= self.num_epochs:
                print("Training finished for this checkpoint")
        else:
            self.last_epoch = 0
            self.min_loss = 1e3

    def _setup_testing(self):
        # Load data
        dataset = SemCorDataset()
        self.data_loader = ElmoSemCorLoader(dataset, batch_size=self.batch_size, win_size=32)
        # Build model
        self.model = SimpleWSD(self.data_loader)
        self.model.to(self.device)

    def train_epoch(self, epoch_i):
        for step, (b_x, b_l, b_y) in enumerate(self.data_loader):
            self.model.zero_grad()
            self.model.h, self.model.cell = map(lambda x: x.to(self.device),
                                                self.model.init_hidden(len(b_y)))

            scores = self.model(b_x.to(self.device), torch.tensor(b_l).to(self.device))
            loss = self.model.loss(scores, b_y, self.device)
            loss.backward()  # compute gradients with back-propagation

            if step % 100 == 0:
                print(f'\rLoss: {loss.item():.4f} ', end='')
                if torch.cuda.is_available():  # check if memory is leaking
                    print(f'Allocated GPU memory: '
                          f'{torch.cuda.memory_allocated() / 1_000_000} MB', end='')
                # possibly save progress
                current_loss = loss.item()
                if current_loss < self.min_loss:
                    min_loss = current_loss
                    torch.save({
                        'epoch': epoch_i,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'current_loss': current_loss,
                        'min_loss': min_loss,
                    }, self.checkpoint_path)
                self.evaluate(epoch_i)
                self.model.train()  # return to train mode

            clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            self.optimizer.step()  # update the weights

    def train(self):
        self.model.train()
        for epoch in range(self.last_epoch + 1, self.num_epochs + 1):
            print(f'\nEpoch: {epoch}')
            self.train_epoch(epoch)

    def evaluate(self,
                 num_epoch,
                 eval_report='logs/baseline_elmo_report.txt',
                 best_model_path='saved_weights/baseline_elmo/best_checkpoint.pt'):

        print("\nEvaluating...", flush=True)
        self.model.eval()
        with torch.no_grad():
            pred, true = [], []
            for step, (b_x, b_str, b_p, b_l, b_y) in enumerate(self.eval_loader):
                self.model.zero_grad()
                self.model.h, self.model.cell = map(lambda x: x.to(self.device),
                                                    self.model.init_hidden(len(b_y)))
                scores = self.model(b_x.to(self.device), torch.tensor(b_l).to(self.device))
                pred += self._select_senses(scores, b_x, b_str, b_p, b_l, b_y)
                true += b_y
            true_flat, pred_flat = [item for sublist in true for item in sublist], \
                                   [item for sublist in pred for item in sublist]
            true_eval, pred_eval = [], []
            for i in range(len(true_flat)):
                if true_flat[i] == 0:
                    continue
                else:
                    true_eval.append(true_flat[i])
                    pred_eval.append(pred_flat[i])
            print(f"True: {true_eval[:25]} ...")
            print(f"Pred: {pred_eval[:25]} ...")
            with open(eval_report, 'w') as fo:
                print(classification_report(
                    true_eval,
                    pred_eval,
                    digits=3),
                    file=fo)
                f1 = f1_score(true_eval, pred_eval, average='micro')
                print(f"F1 = {f1}")

            if f1 > self.best_f1_micro:
                self.best_f1_micro = f1
                torch.save({
                    'epoch': num_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'f1': self.best_f1_micro
                }, best_model_path)

    def test(self,
             eval_report='logs/baseline_elmo_report_test.txt',
             best_model_path='saved_weights/baseline_elmo/best_checkpoint.pt'):
        """
        Evaluate on all test dataset.
        """
        test_dataset = SemCorDataset(data_path='res/wsd-train/test_data.xml',
                                     tags_path='res/wsd-train/test_tags.txt',
                                     sense2id=self.data_loader.dataset.sense2id)
        test_loader = ElmoLemmaPosLoader(test_dataset, batch_size=32, win_size=32)

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Could not find any best model checkpoint.")

        print("\nEvaluating on concatenation of all dataset...", flush=True)
        self.model.eval()
        with torch.no_grad():
            pred, true = [], []
            for step, (b_x, b_str, b_p, b_l, b_y) in enumerate(test_loader):
                self.model.zero_grad()
                self.model.h, self.model.cell = map(lambda x: x.to(self.device),
                                                    self.model.init_hidden(len(b_y)))
                scores = self.model(b_x.to(self.device), torch.tensor(b_l).to(self.device))
                pred += self._select_senses(scores, b_x, b_str, b_p, b_l, b_y)
                true += b_y
            true_flat, pred_flat = [item for sublist in true for item in sublist], \
                                   [item for sublist in pred for item in sublist]
            true_eval, pred_eval = [], []
            for i in range(len(true_flat)):
                if true_flat[i] == 0:
                    continue
                else:
                    true_eval.append(true_flat[i])
                    pred_eval.append(pred_flat[i])
            with open(eval_report, 'w') as fo:
                print(classification_report(
                          true_eval,
                          pred_eval,
                          digits=3),
                      file=fo)
                f1 = f1_score(true_eval, pred_eval, average='micro')
                print(f"F1 = {f1}")

    def _select_senses(self, b_scores, b_vec, b_str, b_pos, b_lengths, b_labels):
        """
        Get the max of scores only of possible senses for a given lemma+pos
        :param b_scores: shape = (batch_s x win_s x sense_vocab_s)
        :param b_vec:
        :param b_str:
        :param b_pos:
        :param b_lengths:
        :return:
        """
        def to_ids(synsets):
            return [self.data_loader.dataset.sense2id[x.name()] for x in synsets]

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
        np.put_along_axis(b_scores, b_impossible_senses, -np.inf, axis=-1)
        return np.argmax(b_scores, -1).tolist()

    def _plot(self, name, value, step):
        if not self._plot_server:
            self._plot_server = SummaryWriter(log_dir='logs')
        self._plot_server.add_scalar(name, value, step)


class WSDTrainerLM(BaseTrainer):

    def __init__(self, learning_rate=0.001, num_epochs=40, batch_size=64,
                 checkpoint_path='saved_weights/baseline_elmo/checkpoint.pt'):
        super().__init__(learning_rate, num_epochs, batch_size, checkpoint_path)
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
            return [self.data_loader.dataset.sense2id[x.name()] for x in synsets]

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
        b_scores = softmax(b_scores, -1)
        b_impossible_senses = np.array(b_impossible_senses)
        np.put_along_axis(b_scores, b_impossible_senses, 0, axis=-1)

        # Update b_scores with geometric mean with language model score.
        for i, sent in enumerate(b_str):
            for k, w in enumerate(sent):
                if b_labels[i][k] != 0:  # i.e. sense tagged word
                    text = ['[CLS]'] + sent + ['[SEP]']
                    text[k+1] = '[MASK]'
                    tokenized_text = []
                    for ww in text:
                        tokenized_text += self.bert_tokenizer.tokenize(ww)
                    masked_index = tokenized_text.index('[MASK]')
                    indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    predictions = self.language_model(tokens_tensor)
                    probabilities = torch.nn.Softmax(dim=0)(predictions[0, masked_index])

                    for S in wn.synsets(w, pos=util.id2wnpos[b_pos[i][k]]):
                        s_id = self.data_loader.dataset.sense2id[S.name()]
                        if S not in self.all_syn_lemmas:
                            self.all_syn_lemmas[S] = get_lemmas(S)
                        syn_tok_ids = []
                        for lemma in self.all_syn_lemmas[S]:
                            tokenized = self.bert_tokenizer.tokenize(lemma)
                            tok_ids = self.bert_tokenizer.convert_tokens_to_ids(tokenized)
                            syn_tok_ids += tok_ids
                        top_k = torch.topk(probabilities[syn_tok_ids, ], k=5)[0].tolist() \
                            if len(syn_tok_ids) > 5 else probabilities[syn_tok_ids, ].tolist()
                        s_score = sum(top_k)
                        b_scores[i, k, s_id] = (b_scores[i, k, s_id] * s_score) ** 0.5

        return np.argmax(b_scores, -1).tolist()


class WSDNetTrainer(BaseTrainer):
    pass
