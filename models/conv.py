import torch.nn as nn


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
    
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.Dropout(0.25),
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, padding=1),
#             nn.Dropout(0.20),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(32*8*8, 500),
#             nn.ReLU(),
#             nn.BatchNorm1d(500),
#             nn.Dropout(p=0.5),
#             nn.Linear(500, 100),
#             nn.ReLU(),
#             nn.Linear(100, 32),
#             nn.ReLU(),
#             nn.Softmax(dim=1)
#         )
#     def forward(self, x):
#         x = self.conv(x)
#         #print(x.size())       # Checking the size of output given out, for updating the input size of first layer in fc.
#         x = x.view(x.size(0), -1)
#         return self.fc(x)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
    
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 32),
            # nn.ReLU(),
            # nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.conv(x)
        #print(x.size())       # Checking the size of output given out, for updating the input size of first layer in fc.
        x = x.view(x.size(0), -1)
        return self.fc(x)

