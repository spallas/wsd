import os

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import optim
from torch.nn.utils import clip_grad_norm_

from data_preprocessing import SemCorDataset, ElmoSemCorLoader, ElmoLemmaPosLoader
from wsd import SimpleWSD

torch.manual_seed(42)
np.random.seed(42)


class BaseTrainer:

    def __init__(self,
                 learning_rate=0.001,
                 num_epochs=20,
                 batch_size=32,
                 checkpoint_path='saved_weights/baseline_elmo/checkpoint.pt'):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        # Using single GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load data
        dataset = SemCorDataset()
        self.data_loader = ElmoSemCorLoader(dataset, batch_size=batch_size, win_size=32)
        eval_dataset = SemCorDataset(data_path='res/wsd-test/se07/se07.xml',
                                     tags_path='res/wsd-test/se07/se07.txt',
                                     sense2id=dataset.sense2id)
        self.eval_loader = ElmoLemmaPosLoader(eval_dataset, batch_size=batch_size,
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
            self.last_epoch = checkpoint['epoch']
            self.min_loss = checkpoint['min_loss']
            print(f"Loaded checkpoint from: {checkpoint_path}")
            if self.last_epoch >= num_epochs:
                print("Training finished for this checkpoint")
        else:
            self.last_epoch = 0
            self.min_loss = 1e3

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

            clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            self.optimizer.step()  # update the weights

    def train(self):
        for epoch in range(self.last_epoch + 1, self.num_epochs + 1):
            print(f'\nEpoch: {epoch}')
            self.train_epoch(epoch)

    def _select_senses(self, b_scores, b_vec, b_str, b_pos, b_lengths):
        return torch.max(b_scores, -1)[1].tolist()  # max[1] are the indices

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
                pred += self._select_senses(scores, b_x, b_str, b_p, b_l)
                true += b_y
            true_eval, pred_eval = [item for sublist in true for item in sublist], \
                                   [item for sublist in pred for item in sublist]
            te, pe = [], []
            for i in range(len(true_eval)):
                if true_eval[i] == 0:
                    continue
                else:
                    te.append(true_eval[i])
                    pe.append(pred_eval[i])
            true_eval, pred_eval = te, pe
            print(f"True: {true_eval}")
            print(f"Pred: {pred_eval}")
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

    def test(self):
        """
        Evaluate on all test dataset.
        """
        pass


class WSDNetTrainer(BaseTrainer):
    pass
