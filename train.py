import os

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from data_preprocessing import SemCorDataset, ElmoSemCorLoader
from wsd import SimpleWSD


def train_elmo():

    learning_rate = 0.001
    checkpoint_path = 'models/baseline_elmo/checkpoint.pt'
    num_epochs = 2
    batch_size = 32

    # Using single GPU
    # noinspection PyUnresolvedReferences
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = SemCorDataset()
    data_loader = ElmoSemCorLoader(dataset, batch_size=batch_size, win_size=32)
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
    else:
        last_epoch = 0
        min_loss = 1e3

    for epoch in range(last_epoch + 1, num_epochs):
        print(f'Epoch: {epoch}')
        for step, (b_x, b_l, b_y) in data_loader:
            model.zero_grad()
            model.h, model.cell = map(lambda x: x.to(device), model.init_hidden(len(b_y)))

            scores = model(b_x.to(device), torch.tensor(b_l).to(device))
            loss = model.loss(scores, b_y, device)
            loss.backward()  # compute gradients with back-propagation

            if step % 100 == 0:
                print(f'\r{loss.item():.4f} ', end='')
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
                    # TODO: evaluate

            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()  # update the weights

    return
