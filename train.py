import os

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import optim
from torch.nn.utils import clip_grad_norm_

from data_preprocessing import SemCorDataset, ElmoSemCorLoader
from wsd import SimpleWSD

torch.manual_seed(42)
np.random.seed(42)


def eval_elmo(model, device, eval_loader, best_f1_micro, epoch, dataset):

    eval_report = 'logs/baseline_elmo_report.txt'
    best_model_path = 'saved_weights/baseline_elmo/best_checkpoint.pt'

    print("\nEvaluating...", flush=True)
    model.eval()
    with torch.no_grad():
        pred, true = [], []
        for step, (b_x, b_l, b_y) in enumerate(eval_loader):
            model.zero_grad()
            model.h, model.cell = map(lambda x: x.to(device), model.init_hidden(len(b_y)))
            scores = model(b_x.to(device), torch.tensor(b_l).to(device))
            pred += torch.max(scores, -1)[1].tolist()
            true += b_y
        print(f"Len true = {len(true)}\nLen pred = {len(pred)}")
        print(f"{pred}")
        print(f"{true}")
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
        with open(eval_report, 'w') as fo:
            print(classification_report(
                        np.array(true_eval),
                        np.array(pred_eval),
                        digits=3),
                  file=fo)
            f1 = f1_score(true_eval, pred_eval, average='micro')
            print(f"F1 = {f1}")

        if f1 > best_f1_micro:
            best_f1_micro = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'f1': best_f1_micro
            }, best_model_path)


def train_elmo():

    learning_rate = 0.001
    checkpoint_path = 'saved_weights/baseline_elmo/checkpoint.pt'
    num_epochs = 2
    batch_size = 32

    # Using single GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = SemCorDataset()
    data_loader = ElmoSemCorLoader(dataset, batch_size=batch_size, win_size=32)
    eval_dataset = SemCorDataset(data_path='res/wsd-test/se07/se07.xml',
                                 tags_path='res/wsd-test/se07/se07.txt')
    eval_loader = ElmoSemCorLoader(eval_dataset, batch_size=batch_size,
                                   win_size=32, overlap_size=8)
    # Build model
    model = SimpleWSD(data_loader)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_f1_micro = 0.0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        print(f"Loaded checkpoint from: {checkpoint_path}")
        if last_epoch >= num_epochs:
            print("Training finished for this checkpoint")
    else:
        last_epoch = 0
        min_loss = 1e3

    for epoch in range(last_epoch + 1, num_epochs + 1):
        print(f'Epoch: {epoch}')
        for step, (b_x, b_l, b_y) in enumerate(data_loader):
            model.zero_grad()
            model.h, model.cell = map(lambda x: x.to(device), model.init_hidden(len(b_y)))

            scores = model(b_x.to(device), torch.tensor(b_l).to(device))
            loss = model.loss(scores, b_y, device)
            loss.backward()  # compute gradients with back-propagation

            if step % 100 == 0:
                print(f'\rLoss: {loss.item():.4f} ', end='')
                if torch.cuda.is_available():  # check if memory is leaking
                    print(f'Allocated GPU memory: {torch.cuda.memory_allocated() / 1_000_000} MB', end='')
                # possibly save progress
                current_loss = loss.item()
                if current_loss < min_loss:
                    min_loss = current_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'current_loss': current_loss,
                        'min_loss': min_loss,
                    }, checkpoint_path)
                eval_elmo(model, device, eval_loader, best_f1_micro, epoch, dataset)

            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()  # update the weights
    return
