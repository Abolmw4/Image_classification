from torch import nn



class NeuralNetwork(nn.Module):
    def __init__(self, kernel_size=3):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels=16, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(16, out_channels=32, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(32, out_channels=64, kernel_size=kernel_size)
        self.activation = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.6)
        self.maxpoling = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.dens1 = nn.Linear(57600, 128)
        self.dens2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpoling(x)
        x = self.activation(x)
        x = self.conv2(x)
        # x = self.batchnorm1(x)
        x = self.maxpoling(x)

        x = self.activation(x)
        x = self.conv3(x)
        # x = self.batchnorm2(x)
        x = self.maxpoling(x)

        x = self.activation(x)
        x = self.flatten(x)
        x = self.dens1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.dens2(x)
        return x