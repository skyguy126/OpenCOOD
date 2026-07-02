import torch.nn as nn


class SimplePlanningHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 mlp_hidden_dim: int = 256,
                 num_waypoints: int = 6):
        super(SimplePlanningHead, self).__init__()
        self.num_waypoints = num_waypoints

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, num_waypoints * 2),
        )

    def forward(self, spatial_features_2d):
        x = self.conv(spatial_features_2d)
        x = self.mlp(x)
        return x.view(x.shape[0], self.num_waypoints, 2)
