import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataloader import ColligatePotentialDataset
from model import create_model


def train(dataset, batch_size = 4, epochs = 10000, lr = 1e-4, train_split = 0.9):
    writer = SummaryWriter(log_dir=f'../logs/{dataset.type}-model_combine={dataset.include_combine}', flush_secs=20)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 1)

    model = create_model(dataset).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        training_loss = 0
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            loss = _calc_loss(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(train_loader)
        writer.add_scalar('Loss/train', training_loss, epoch)

        if epoch == 0 or epoch % 100 == 99:
            with torch.no_grad():
                model.eval()
                valid_loss = 0
                for batch in test_loader:
                    loss = _calc_loss(model, batch, criterion, device)
                    valid_loss += loss
                valid_loss /= len(test_loader)
                if valid_loss >= best_loss: break
                best_loss = valid_loss
                torch.save(model.state_dict(), f'../models/{dataset.type}-model_combine={dataset.include_combine}.pt')
                writer.add_scalar('Loss/test', valid_loss, epoch)


def _calc_loss(model, batch, criterion, device):
    features = {k: v.to(device) for k, v in batch['features'].items()}
    predictions = model(**features)
    labels = batch['labels'].unsqueeze(-1).to(device)
    loss = criterion(predictions, labels)
    return loss
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='passing, rushing, receiving')
    parser.add_argument('--include_combine', action='store_true', help='include combine stats')
    parser.add_argument('--batch_size', type=int, default=4, help='size of training batch')
    parser.add_argument('--epochs', type=int, default=10000, help='num training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizers initial learning rate')
    parser.add_argument('--train_split', type=float, default=0.9, help='prop of data to use for training')
    args = parser.parse_args()
    dataset = ColligatePotentialDataset(type=args.type, include_combine=args.include_combine)
    train(dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, train_split=args.train_split)
