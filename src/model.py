import torch.nn as nn
import torch

class SkillPredictor(nn.Module):

    def __init__(self, college_stats_size, schools_size, combine_stats_size = None):
        super().__init__()
        self.college_stats_size = college_stats_size
        self.schools_size = schools_size
        self.combine_stats_size = combine_stats_size

        self.school_hidden_size = 32
        self.school_output_size = 1
        self.school_hidden_layer = nn.Linear(self.schools_size, self.school_hidden_size)
        self.school_output_layer = nn.Linear(self.school_hidden_size, self.school_output_size)

        self.layer1 = nn.Linear(self.school_output_size + college_stats_size + combine_stats_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    
    def forward(self, college_stats, combine_stats, school_encoding):
        school_score = self.school_output_layer(torch.sigmoid(self.school_hidden_layer(school_encoding)))
        x = torch.cat((school_score, college_stats, combine_stats), dim = 1)
        # nomalize x
        x = x / torch.norm(x, p = 2, dim = 1).unsqueeze(1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return x