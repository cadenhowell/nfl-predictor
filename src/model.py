import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(dataset):
    '''
    Given a dataset, extracts feature sizes to create a model.
    '''
    example = dataset[0]['features']
    stats = [example['college_stats'], example['school_encoding']]
    if dataset.include_combine:
        stats.append(example['combine_stats'])
    sizes = map(lambda stat: stat.shape[-1], stats)
    model = SkillPredictor(*sizes)
    return model


class SkillPredictor(nn.Module):

    def __init__(self, college_stats_size, schools_size, combine_stats_size = None):
        super().__init__()
        self.college_stats_size = college_stats_size
        self.schools_size = schools_size
        self.combine_stats_size = 0 if combine_stats_size is None else combine_stats_size

        school_output_size = 1
        self.school_model = nn.Sequential(
            nn.Linear(self.schools_size, 16),
            nn.ReLU(),
            nn.Linear(16, school_output_size),
            nn.ReLU()
        )

        main_input_size = self.college_stats_size + self.combine_stats_size + school_output_size
        layer1_size = 2 * main_input_size // 3
        self.prediction_model = nn.Sequential(
            nn.Linear(main_input_size, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, 1)
        )

    
    def forward(self, **kwargs):
        school_score = self.school_model(kwargs['school_encoding'])
        x = torch.cat((school_score, kwargs['college_stats']), dim=-1)
        if 'combine_stats' in kwargs:
            x = torch.cat((x, kwargs['combine_stats']), dim=-1)
        x = F.normalize(x, p = 2, dim = -1)
        return self.prediction_model(x)
