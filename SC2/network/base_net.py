import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        q = self.fc2(h)
        return q, h


class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.layer = nn.Sequential(
            nn.Linear(input_shape, args.critic_dim),
            nn.ReLU(),
            nn.Linear(args.critic_dim, args.critic_dim),
            nn.ReLU(),
            nn.Linear(args.critic_dim, 1)
        )

    def forward(self, inputs):
        return self.layer(inputs)


class Reward_net(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(Reward_net, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))

        h_in = hidden_state.reshape(-1, self.hidden_dim)[0]
        h = self.rnn(x, h_in)

        r = self.fc2(h)
        return r, h
