import torch.nn as nn
import torch


class QStar(nn.Module):
    def __init__(self, args):
        super(QStar, self).__init__()
        self.args = args
        self.input_shape = args.state_shape + self.args.n_agents
        self.fc = nn.Sequential(nn.Linear(self.input_shape, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1))

    def forward(self, qsas, states):
        episode_num = qsas.size(0)
        qsas = qsas.view(-1, self.args.n_agents)
        states = states.reshape(-1, self.args.state_shape)

        input = torch.cat((qsas, states), 1)

        q_star = self.fc(input)
        q_star = q_star.view(episode_num, -1, 1)
        return q_star
