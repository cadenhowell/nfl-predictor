import argparse

import torch

from dataloader import ColligatePotentialDataset
from model import create_model


def predict(name, type, include_combine):
    dataset = ColligatePotentialDataset(type=type, include_combine=include_combine)

    model = create_model(dataset)
    try:
        print(f'../models/{type}_model-combine={include_combine}.pt')
        model.load_state_dict(torch.load(f'../models/{type}-model_combine={include_combine}.pt'))
    except FileNotFoundError:
        print(f'No model found. Run python train.py {type} {"--include_combine" if include_combine else ""}')
        exit(1)
    
    model.eval()

    try:
        player_stats = dataset.get_player_stats(name)
    except KeyError:
        print(f'Player {name} not found')
        exit(1)

    prediction = model(**player_stats['features'])

    print(f'Player: {name}')
    print(f'Prediction: {prediction.item()}')
    print(f'Actual: {player_stats["labels"].item()}')
    print(f'Error: {abs(prediction.item() - player_stats["labels"].item())}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='name of the player to predict')
    parser.add_argument('type', type=str, help='passing, rushing, receiving')
    parser.add_argument('--include_combine', action='store_true', help='include combine stats')
    args = parser.parse_args()
    predict(args.name, args.type, args.include_combine)
