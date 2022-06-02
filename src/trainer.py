import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataloader import ColligatePotentialDataset
from model import SkillPredictor


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ColligatePotentialDataset('passing', include_combine=True)

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
    model = SkillPredictor(example[0].shape[-1], example[2].shape[-1], example[1].shape[-1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    for epoch in range(max_epochs):
        print(epoch)
        for college_stats, combine_stats, school_encoding, madden_rating in train_loader:
            college_stats, combine_stats, school_encoding, madden_rating = college_stats.to(device), combine_stats.to(device), school_encoding.to(device), madden_rating.to(device)
            optimizer.zero_grad()
            predictions = model(college_stats, combine_stats, school_encoding)
            loss = criterion(predictions, madden_rating)
            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            mse_loss = 0
            for college_stats, combine_stats, school_encoding, madden_rating in test_loader:
                college_stats, combine_stats, school_encoding, madden_rating = college_stats.to(device), combine_stats.to(device), school_encoding.to(device), madden_rating.to(device)
                predictions = model(college_stats, combine_stats, school_encoding)
                loss = criterion(predictions, madden_rating)
                mse_loss += loss
            print(f'Avg MSE Loss: {mse_loss / len(test_loader)}')


if __name__ == '__main__':
    train()
