import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1,hidden_dim_2,num_classes, num_layers=1):
        super(SequenceClassifier, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.lstm_1 = nn.LSTM(input_dim, hidden_dim_1, num_layers=num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_dim_1, hidden_dim_2, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim_2, num_classes)

    def forward(self, x):
        
        lstm_out, _ = self.lstm_1(x) #passing the embeddings through lstm layer
        dropout1 = self.dropout(lstm_out)
        
        lstm_out, _ = self.lstm_2(dropout1)
        lstm_out = lstm_out[:, -1, :]

        output = self.fc(lstm_out) # Taking the output from the last time step and passing through Linear layer to map to 4 classes.

        output_probs = torch.softmax(output, dim=1)
        return output_probs

