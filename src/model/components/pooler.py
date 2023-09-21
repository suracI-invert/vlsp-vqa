from torch import nn

class Pooler(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.activation = nn.Tanh()

    # def forward(self, hidden_states):
