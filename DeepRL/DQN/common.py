import torch.nn as nn 
class DuelMlpPolicy(nn.Module):
    def __init__(self, n_observations,output_dim):
        super().__init__()
        self.state_value = nn.Sequential(
              nn.Linear(in_features = n_observations, out_features = 512),
              nn.ReLU(),
              nn.Linear(in_features=512, out_features=256),
              nn.ReLU(),
              nn.Linear(in_features=256, out_features=1),
            )
        self.action_value = nn.Sequential(
              nn.Linear(in_features = n_observations, out_features = 512),
              nn.ReLU(),
              nn.Linear(in_features=512, out_features=256),
              nn.ReLU(),
              nn.Linear(in_features=256, out_features=output_dim),
            )
    def forward(self,x):
        action_value = self.action_value(x)
        state_value = self.state_value(x)
        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        q = state_value + action_score_centered
        return q

class DuelCnnPolicy(nn.Module):
    def __init__(self, c,output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.state_value = nn.Sequential(
              nn.Linear(in_features=512, out_features=256),
              nn.ReLU(),
              nn.Linear(in_features=256, out_features=1),
            )
        self.action_value = nn.Sequential(
              nn.Linear(in_features=512, out_features=256),
              nn.ReLU(),
              nn.Linear(in_features=256, out_features=output_dim),
            )
    def forward(self,x):
        x = self.net(x)
        action_value = self.action_value(x)
        state_value = self.state_value(x)
        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        q = state_value + action_score_centered
        return q

class DqnMlpPolicy(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_observations, 128),
                                 nn.Linear(128,128),
                                 nn.Linear(128,n_actions))
    def forward(self,x):
        return self.net(x)


class DqnCnnPolicy(nn.Module): 
    def __init__(self, c,output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=output_dim),
        )
    def forward(self,x):
        return self.net(x)
    
