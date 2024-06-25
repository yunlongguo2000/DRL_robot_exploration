import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    def __init__(self, num_actions):
        super(CNNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMNetwork(nn.Module):
    def __init__(self, num_actions, num_cells):
        super(LSTMNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 512, kernel_size=7, stride=1)
        self.lstm = nn.LSTMCell(512, num_cells)
        self.fc = nn.Linear(num_cells, num_actions)

    def forward(self, x, trace_length, batch_size, state):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), 512)
        outputs = []
        for i in range(trace_length):
            state = self.lstm(x[i * batch_size:(i + 1) * batch_size], state)
            output = self.fc(state[0])
            outputs.append(output)
        outputs = torch.cat(outputs, 0)
        return outputs, state