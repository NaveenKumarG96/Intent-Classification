# IntentClassification

Install the required dependencies into the virtual environment

```pip install -r requirements.txt```

Run the uvicorn standard command to run the app.py --- to run the model with lstm classifier

```uvicorn app:app --host 0.0.0.0 --port 8000 --reload```  

Run the uvicorn standard command to run the appRoberta.py --- to run the model with lstm classifier

```uvicorn app:appRoberta --host 0.0.0.0 --port 8000 --reload``` 

Use Postman POST api to check the intent of the statment -  use the below url for post action.
```http://0.0.0.0:8000/intent```

Also can give the query using the browser

```http://0.0.0.0:8000/```

# Training the model

```cd training```

```python3 training_code.py```

class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SequenceClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) #passing the embeddings through lstm layer
        lstm_out = lstm_out[:, -1, :]
        dropout = self.dropout(lstm_out)
        lstm_out, _ = self.lstm(dropout)
        lstm_out = lstm_out[:, -1, :]

 # Taking the output from the last time step and passing through Linear layer to map to 4 classes.
        output = self.fc(lstm_out)

        output_probs = torch.softmax(output, dim=1)
        return output