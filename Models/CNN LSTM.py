class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()  # ((n+ 2p -f) / st ) +1  # 16
        self.ff= nn.Linear(12,1728) # 12**3
        self.conv1 = nn.Conv1d(8164, 1024, kernel_size=3, stride=1, padding=1)   #18
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # 2  # b2t 9
        self.conv2 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1)   #  9
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) #  Floor((n-f+2p)/s) +1  #4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4, 128) # 4 --> sequence length //4(34an 3aml 2 maxpool kol wa7da b 2)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 14)  # Output


    def forward(self, x):
        x= self.ff(x)
        #x.size(0)
        x = x.view(-1, 8164, 18)  # 18*8164 = 12* 12246
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
import torch
import torch.nn as nn

class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        # Apply the fully connected layer to each time step
        out = self.fc(out)

        return out


# Define the input and target dimensions
input_dim = 14
output_dim = 14
hidden_dim = 64
num_layers = 1
batch_size = 50


import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, cnn_model, lstm_model):
        super(CombinedModel, self).__init__()
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model

    def forward(self, x):
        # Forward pass through CNN
        cnn_output = self.cnn_model(x)

        # Reshape CNN output to fit LSTM input shape
        cnn_output = cnn_output.view(cnn_output.size(0), -1, cnn_output.size(-1))

        # Forward pass through LSTM using CNN output
        lstm_output = self.lstm_model(cnn_output)

        return lstm_output

# Create instances of your CNN and LSTM models
cnn_model = CNN()
lstm_model = MyLSTMModel(input_dim, hidden_dim, num_layers, output_dim)

if torch.cuda.is_available():
    cnn_model = cnn_model.to('cuda')
    lstm_model = lstm_model.to('cuda')

# Create the combined model
combined_model = CombinedModel(cnn_model, lstm_model)

# Print the combined model architecture
print(combined_model)
