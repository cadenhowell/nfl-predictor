import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import create_model 
from dataloader import ColligatePotentialDataset


def train(dataset, batch_size = 8, epochs = 100, lr = 1e-2, train_split = 0.8):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset)

    model = create_model(dataset).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        print(epoch)
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            loss = _calc_loss(model, batch, criterion, device)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            mse_loss = 0
            for batch in test_loader:
                loss = _calc_loss(model, batch, criterion, device)
                mse_loss += loss
            mse_loss /= len(test_loader)
            if mse_loss < best_loss:
                best_loss = mse_loss
                torch.save(model.state_dict(), f'../models/{dataset.type}-model_combine={dataset.include_combine}.pt')
            print(f'Avg MSE Loss: {mse_loss}')
    print(f'Best MSE Loss: {best_loss}')


def _calc_loss(model, batch, criterion, device):
    features = {k: v.to(device) for k, v in batch['features'].items()}
    predictions = model(**features)
    loss = criterion(predictions, batch['labels'].to(device))
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='passing, rushing, receiving')
    parser.add_argument('--include_combine', action='store_true', help='include combine stats')
    parser.add_argument('--batch_size', type=int, default=4, help='size of training batch')
    parser.add_argument('--epochs', type=int, default=100, help='num training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='optimizers initial learning rate')
    parser.add_argument('--train_split', type=float, default=0.1, help='prop of data to use for training')
    args = parser.parse_args()
    dataset = ColligatePotentialDataset(type=args.type, include_combine=args.include_combine)
    train(dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, train_split=args.train_split)
