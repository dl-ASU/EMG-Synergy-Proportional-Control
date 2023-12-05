class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()   # ((n+ 2p -f) / st ) +1  # 16
        self.ff= nn.Linear(12,1728) # 12**3
        # self.X= X.reshape((-1, 100, 16))
        self.conv1 = nn.Conv1d(108, 256, kernel_size=3, stride=1, padding=1)  # 32 ll input channels | 64 ll output channels #16
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # 2  # b2t 8
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1) # 64 input channels | 128 output channels   #  8
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) #2   #4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4, 128) # 4 --> sequence length //4(34an 3aml 2 maxpool kol wa7da b 2)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 14)  # Output


    def forward(self, x):
        x= self.ff(x)
        x= x.reshape((-1,108,16))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
