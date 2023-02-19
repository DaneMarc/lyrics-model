import torch.nn as nn
import torch.nn.functional as F

class DeathStar(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self._batch_size = hparams.batch_size
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3) 
        self.fc3 = nn.Linear(3, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
