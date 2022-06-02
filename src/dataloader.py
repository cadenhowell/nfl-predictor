import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _get_valid_pos(type):
    match type:
        case 'passing':
            return ['QB']
        case 'rushing':
            return ['RB', 'HB', 'FB']
        case 'receiving':
            return ['WR', 'TE']


def _load_player_stats(type):
    '''
    Loads and returns a DataFrame of players' college stats
    '''
    df = pd.read_csv(f'../data/college-stats/college_{type}_stats.csv')
    df.drop_duplicates(subset=['Player', 'Year'], keep=False, inplace=True)
    df.set_index('Player', inplace=True)
    df = df[df.index.value_counts().lt(5)]
    schools = pd.get_dummies(df['School'])
    df.drop(columns=['Conf', 'Year', 'School'], inplace=True)
    return df, schools


def _load_combine_stats(type):
    '''
    Loads and returns a DataFrame of NFL Combine stats relevant to the given position type
    '''
    df = pd.read_csv(f'../data/combine-stats/combine_data.csv')
    valid_pos = _get_valid_pos(type)
    df = df[df['Pos'].isin(valid_pos)]
    df.drop_duplicates(subset=['Player'], keep=False, inplace=True)
    df.set_index('Player', inplace=True)
    df.drop(columns=['Year', 'Pos', 'School', 'Team', 'Round', 'Pick', 'Draftyear'], inplace=True)
    return df


def _load_madden_stats(type):
    '''
    Loads and returns a DataFrame of Madden overall labels relevant to the given type. 
    For players appearing in several years, the overalls are averaged.
    '''
    valid_pos = _get_valid_pos(type)
    files = glob.glob(os.path.join('../data/madden-stats', "*.csv"))
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df = df[df['Position'].isin(valid_pos)]
        if 'FirstName' in df.columns:
            df['FirstName'] = df['FirstName'] + ' ' + df['LastName']
        elif 'First Name' in df.columns:
            df['First Name'] = df['First Name'] + ' ' + df['Last Name']
        first_name_headers = ('FirstName', 'First Name', 'Full Name', 'Name')
        first_name_dict = {name: 'Player' for name in first_name_headers}
        overall_headers = ('OVR', 'Overall Rating', 'OverallRating')
        overall_dict = {header: 'Overall' for header in overall_headers}
        rename_dict = first_name_dict | overall_dict
        df.rename(columns=rename_dict, inplace=True)
        df = df[['Player', 'Overall']]
        df.drop_duplicates(subset=['Player'], keep=False, inplace=True)
        dfs.append(df)
    stats = pd.concat(dfs)
    return stats.groupby(['Player']).median()


def _get_players(college_stats, madden_stats, combine_stats = None):
    '''
    Returns an Index of players who appear in all datasets
    '''
    players = college_stats.index
    players &= madden_stats.index
    if combine_stats is not None:
        players &= combine_stats.index
    return players.unique()


class ColligatePotentialDataset(Dataset):
    '''
    Loads passing, rushing, or receiving data into a Dataset object.
    '''
    def __init__(self, type = 'passing', include_combine = True):
        if type not in ['passing', 'rushing', 'receiving']:
            raise ValueError('type must be one of "passing", "rushing", "receiving"')
        self.type = type
        self.include_combine = include_combine

        all_college_stats, all_schools = _load_player_stats(type)
        all_madden_stats = _load_madden_stats(type)
        stats = [all_college_stats, all_madden_stats]
        if include_combine:
            all_combine_stats = _load_combine_stats(type)
            stats.append(all_combine_stats)

        self.players = _get_players(*stats)

        self.college_stats = all_college_stats.loc[self.players]
        self.schools = all_schools.loc[self.players].drop_duplicates().groupby(['Player']).sum()
        self.madden_stats = all_madden_stats.loc[self.players]
        if include_combine:
            self.combine_stats = all_combine_stats.loc[self.players]

    def __len__(self):
        return len(self.players)

    def __getitem__(self, idx):
        '''
        Returns a tensor of the player's college stats, NLF Combine stats, school, and Madden rating in that order.
        If include_combine is False, only returns college stats, school, and Madden rating in that order.
        '''
        items = []

        player = self.players[idx]

        college_stats = self.college_stats.loc[player].values
        four_season_length = 4 * len(college_stats[0])
        flat_college_stats = college_stats.flatten()
        padding = np.zeros(four_season_length - len(flat_college_stats))
        padded_college_stats = np.append(padding, flat_college_stats)
        items.append(torch.tensor(padded_college_stats))

        if self.include_combine:
            combine_stats = self.combine_stats.loc[player].values
            items.append(torch.tensor(combine_stats))

        school = self.schools.loc[player].values
        items.append(torch.tensor(school))

        madden_rating = self.madden_stats.loc[player]['Overall']
        items.append(torch.tensor(madden_rating))
        
        return items