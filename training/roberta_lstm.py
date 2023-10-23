from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch.nn as nn

num_labels = 4
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)


class LSTMClassifier(nn.Module):
    def __init__(self, model, hidden_size, num_layers, num_labels):
        super(LSTMClassifier, self).__init__()
        self.roberta = model.roberta
        self.lstm = nn.LSTM(input_size=model.config.hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(hidden_states)
        
        logits = self.fc(lstm_out[:, -1, :])
        probas = self.softmax(logits)

        return probas
