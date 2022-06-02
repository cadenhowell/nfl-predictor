import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataloader import ColligatePotentialDataset
from model import SkillPredictor


def train(type, include_combine):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ColligatePotentialDataset(type, include_combine)

    batch_size = 1
    train_split = 0.8
    shuffle_dataset = True
    max_epochs = 100

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset)
    test_loader = DataLoader(test_dataset)

    example = dataset[0]
    model = SkillPredictor(example[0].shape[-1], example[1].shape[-1], example[2].shape[-1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(max_epochs):
        print(epoch)
        for stats in train_loader:
            stats = [stat.to(device) for stat in stats]
            optimizer.zero_grad()
            predictions = model(*stats[:3])
            loss = criterion(predictions, stats[-1])
            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            mse_loss = 0
            for stats in test_loader:
                stats = [stat.to(device) for stat in stats]
                predictions = model(*stats[:3])
                loss = criterion(predictions, stats[-1])
                mse_loss += loss
            mse_loss /= len(test_loader)
            if mse_loss < best_loss:
                best_loss = mse_loss
                torch.save(model.state_dict(), f'../models/{type}_model_combine={include_combine}.pt')
            print(f'Avg MSE Loss: {mse_loss}')
    print(f'Best MSE Loss: {best_loss}')

def predict(player, type, include_combine):
    dataset = ColligatePotentialDataset(type, include_combine)
    player_index = dataset.players.get_loc(player)
    player_stats = dataset[player_index]
    example = dataset[0]
    model = SkillPredictor(example[0].shape[-1], example[1].shape[-1], example[2].shape[-1])
    model.load_state_dict(torch.load(f'../models/{type}_model_combine={include_combine}.pt'))
    predictions = model(*player_stats[:3])
    truth = player_stats[-1]
    print(f'Predictions: {predictions}')
    print(f'Truth: {truth}')


if __name__ == '__main__':
    type = 'passing'
    include_combine = True
    # train(type, include_combine)
    predict('Aaron Rodgers', type, include_combine)
