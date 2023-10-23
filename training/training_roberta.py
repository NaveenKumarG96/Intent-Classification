import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
from lstm import LSTMClassifier
train_data_path = '../data/train.json'

test_data_path = '../data/test.json'

def load_data(data_path):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    return data

train_data = load_data(train_data_path)

test_data = load_data(test_data_path)

label_mapping = {"Churn": 0, "Escalation": 1,  'Churn and Escalation':2,"No Intent Found": 3}

# Tokenize and encode data
def preprocess_data(data, tokenizer, label_mapping):
    input_texts = [example["text"] for example in data]
    labels = [label_mapping[example["intent"]] for example in data]

    tokenized_data = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]

    return input_ids, attention_mask, labels

# Load the RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
num_labels = len(label_mapping)
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

# Preprocess training and test data
train_input_ids, train_attention_mask, train_labels = preprocess_data(train_data, tokenizer, label_mapping)
test_input_ids, test_attention_mask, test_labels = preprocess_data(test_data, tokenizer, label_mapping)

# Create PyTorch datasets
class IntentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
    

train_dataset = IntentDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = IntentDataset(test_input_ids, test_attention_mask, test_labels)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize and train the RoBERTa model

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()




# Initialize the LSTM classifier
hidden_size = 64
num_layers = 1
lstm_classifier = LSTMClassifier(model, hidden_size, num_layers, num_labels)
# lstm_classifier.load_state_dict(model.state_dict(), strict=False)

# model.eval()
num_epochs = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_classifier.to(device)
lstm_classifier.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
    
        optimizer.zero_grad()
        logits = lstm_classifier(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

model_int8 = torch.quantization.quantize_dynamic(
    lstm_classifier,
    {torch.nn.Linear},  
    dtype=torch.qint8)

# torch.save(model,'../data/roberta.pth') 

state_dict = model_int8.state_dict()
attribute_names = list(state_dict.keys())
print(attribute_names)
# print(attribute_names)

print()
attributes = vars(lstm_classifier)
# print(attributes)
selected_attributes = {}

attributes_to_save = [ 'roberta.embeddings.position_embeddings.weight', 'roberta.embeddings.token_type_embeddings.weight', 'roberta.embeddings.LayerNorm.weight', 'roberta.embeddings.LayerNorm.bias','roberta.encoder.layer.0.attention.self.query.scale', 'roberta.encoder.layer.0.attention.self.query.zero_point', 'roberta.encoder.layer.0.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.0.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.0.attention.self.key.scale', 'roberta.encoder.layer.0.attention.self.key.zero_point', 'roberta.encoder.layer.0.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.0.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.0.attention.self.value.scale', 'roberta.encoder.layer.0.attention.self.value.zero_point', 'roberta.encoder.layer.0.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.0.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.0.attention.output.dense.scale', 'roberta.encoder.layer.0.attention.output.dense.zero_point', 'roberta.encoder.layer.0.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.0.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.0.attention.output.LayerNorm.weight', 'roberta.encoder.layer.0.attention.output.LayerNorm.bias', 'roberta.encoder.layer.0.intermediate.dense.scale', 'roberta.encoder.layer.0.intermediate.dense.zero_point', 'roberta.encoder.layer.0.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.0.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.0.output.dense.scale', 'roberta.encoder.layer.0.output.dense.zero_point', 'roberta.encoder.layer.0.output.dense._packed_params.dtype', 'roberta.encoder.layer.0.output.dense._packed_params._packed_params', 'roberta.encoder.layer.0.output.LayerNorm.weight', 'roberta.encoder.layer.0.output.LayerNorm.bias', 'roberta.encoder.layer.1.attention.self.query.scale', 'roberta.encoder.layer.1.attention.self.query.zero_point', 'roberta.encoder.layer.1.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.1.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.1.attention.self.key.scale', 'roberta.encoder.layer.1.attention.self.key.zero_point', 'roberta.encoder.layer.1.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.1.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.1.attention.self.value.scale', 'roberta.encoder.layer.1.attention.self.value.zero_point', 'roberta.encoder.layer.1.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.1.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.1.attention.output.dense.scale', 'roberta.encoder.layer.1.attention.output.dense.zero_point', 'roberta.encoder.layer.1.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.1.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.1.attention.output.LayerNorm.weight', 'roberta.encoder.layer.1.attention.output.LayerNorm.bias', 'roberta.encoder.layer.1.intermediate.dense.scale', 'roberta.encoder.layer.1.intermediate.dense.zero_point', 'roberta.encoder.layer.1.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.1.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.1.output.dense.scale', 'roberta.encoder.layer.1.output.dense.zero_point', 'roberta.encoder.layer.1.output.dense._packed_params.dtype', 'roberta.encoder.layer.1.output.dense._packed_params._packed_params', 'roberta.encoder.layer.1.output.LayerNorm.weight', 'roberta.encoder.layer.1.output.LayerNorm.bias','roberta.encoder.layer.2.attention.self.query.scale', 'roberta.encoder.layer.2.attention.self.query.zero_point', 'roberta.encoder.layer.2.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.2.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.2.attention.self.key.scale', 'roberta.encoder.layer.2.attention.self.key.zero_point', 'roberta.encoder.layer.2.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.2.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.2.attention.self.value.scale', 'roberta.encoder.layer.2.attention.self.value.zero_point', 'roberta.encoder.layer.2.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.2.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.2.attention.output.dense.scale', 'roberta.encoder.layer.2.attention.output.dense.zero_point', 'roberta.encoder.layer.2.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.2.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.2.attention.output.LayerNorm.weight', 'roberta.encoder.layer.2.attention.output.LayerNorm.bias', 'roberta.encoder.layer.2.intermediate.dense.scale', 'roberta.encoder.layer.2.intermediate.dense.zero_point', 'roberta.encoder.layer.2.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.2.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.2.output.dense.scale', 'roberta.encoder.layer.2.output.dense.zero_point', 'roberta.encoder.layer.2.output.dense._packed_params.dtype', 'roberta.encoder.layer.2.output.dense._packed_params._packed_params', 'roberta.encoder.layer.2.output.LayerNorm.weight', 'roberta.encoder.layer.2.output.LayerNorm.bias', 'roberta.encoder.layer.3.attention.self.query.scale', 'roberta.encoder.layer.3.attention.self.query.zero_point', 'roberta.encoder.layer.3.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.3.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.3.attention.self.key.scale', 'roberta.encoder.layer.3.attention.self.key.zero_point', 'roberta.encoder.layer.3.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.3.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.3.attention.self.value.scale', 'roberta.encoder.layer.3.attention.self.value.zero_point', 'roberta.encoder.layer.3.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.3.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.3.attention.output.dense.scale', 'roberta.encoder.layer.3.attention.output.dense.zero_point', 'roberta.encoder.layer.3.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.3.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.3.attention.output.LayerNorm.weight', 'roberta.encoder.layer.3.attention.output.LayerNorm.bias', 'roberta.encoder.layer.3.intermediate.dense.scale', 'roberta.encoder.layer.3.intermediate.dense.zero_point', 'roberta.encoder.layer.3.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.3.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.3.output.dense.scale', 'roberta.encoder.layer.3.output.dense.zero_point', 'roberta.encoder.layer.3.output.dense._packed_params.dtype', 'roberta.encoder.layer.3.output.dense._packed_params._packed_params', 'roberta.encoder.layer.3.output.LayerNorm.weight', 'roberta.encoder.layer.3.output.LayerNorm.bias', 'roberta.encoder.layer.4.attention.self.query.scale', 'roberta.encoder.layer.4.attention.self.query.zero_point', 'roberta.encoder.layer.4.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.4.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.4.attention.self.key.scale', 'roberta.encoder.layer.4.attention.self.key.zero_point', 'roberta.encoder.layer.4.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.4.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.4.attention.self.value.scale', 'roberta.encoder.layer.4.attention.self.value.zero_point', 'roberta.encoder.layer.4.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.4.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.4.attention.output.dense.scale', 'roberta.encoder.layer.4.attention.output.dense.zero_point', 'roberta.encoder.layer.4.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.4.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.4.attention.output.LayerNorm.weight', 'roberta.encoder.layer.4.attention.output.LayerNorm.bias', 'roberta.encoder.layer.4.intermediate.dense.scale', 'roberta.encoder.layer.4.intermediate.dense.zero_point', 'roberta.encoder.layer.4.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.4.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.4.output.dense.scale', 'roberta.encoder.layer.4.output.dense.zero_point', 'roberta.encoder.layer.4.output.dense._packed_params.dtype', 'roberta.encoder.layer.4.output.dense._packed_params._packed_params', 'roberta.encoder.layer.4.output.LayerNorm.weight', 'roberta.encoder.layer.4.output.LayerNorm.bias', 'roberta.encoder.layer.5.attention.self.query.scale', 'roberta.encoder.layer.5.attention.self.query.zero_point', 'roberta.encoder.layer.5.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.5.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.5.attention.self.key.scale', 'roberta.encoder.layer.5.attention.self.key.zero_point', 'roberta.encoder.layer.5.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.5.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.5.attention.self.value.scale', 'roberta.encoder.layer.5.attention.self.value.zero_point', 'roberta.encoder.layer.5.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.5.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.5.attention.output.dense.scale', 'roberta.encoder.layer.5.attention.output.dense.zero_point', 'roberta.encoder.layer.5.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.5.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.5.attention.output.LayerNorm.weight', 'roberta.encoder.layer.5.attention.output.LayerNorm.bias', 'roberta.encoder.layer.5.intermediate.dense.scale', 'roberta.encoder.layer.5.intermediate.dense.zero_point', 'roberta.encoder.layer.5.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.5.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.5.output.dense.scale', 'roberta.encoder.layer.5.output.dense.zero_point', 'roberta.encoder.layer.5.output.dense._packed_params.dtype', 'roberta.encoder.layer.5.output.dense._packed_params._packed_params', 'roberta.encoder.layer.5.output.LayerNorm.weight', 'roberta.encoder.layer.5.output.LayerNorm.bias','roberta.encoder.layer.6.attention.self.query.scale', 'roberta.encoder.layer.6.attention.self.query.zero_point', 'roberta.encoder.layer.6.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.6.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.6.attention.self.key.scale', 'roberta.encoder.layer.6.attention.self.key.zero_point', 'roberta.encoder.layer.6.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.6.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.6.attention.self.value.scale', 'roberta.encoder.layer.6.attention.self.value.zero_point', 'roberta.encoder.layer.6.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.6.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.6.attention.output.dense.scale', 'roberta.encoder.layer.6.attention.output.dense.zero_point', 'roberta.encoder.layer.6.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.6.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.6.attention.output.LayerNorm.weight', 'roberta.encoder.layer.6.attention.output.LayerNorm.bias', 'roberta.encoder.layer.6.intermediate.dense.scale', 'roberta.encoder.layer.6.intermediate.dense.zero_point', 'roberta.encoder.layer.6.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.6.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.6.output.dense.scale', 'roberta.encoder.layer.6.output.dense.zero_point', 'roberta.encoder.layer.6.output.dense._packed_params.dtype', 'roberta.encoder.layer.6.output.dense._packed_params._packed_params', 'roberta.encoder.layer.6.output.LayerNorm.weight', 'roberta.encoder.layer.6.output.LayerNorm.bias', 'roberta.encoder.layer.7.attention.self.query.scale', 'roberta.encoder.layer.7.attention.self.query.zero_point', 'roberta.encoder.layer.7.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.7.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.7.attention.self.key.scale', 'roberta.encoder.layer.7.attention.self.key.zero_point', 'roberta.encoder.layer.7.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.7.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.7.attention.self.value.scale', 'roberta.encoder.layer.7.attention.self.value.zero_point', 'roberta.encoder.layer.7.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.7.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.7.attention.output.dense.scale', 'roberta.encoder.layer.7.attention.output.dense.zero_point', 'roberta.encoder.layer.7.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.7.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.7.attention.output.LayerNorm.weight', 'roberta.encoder.layer.7.attention.output.LayerNorm.bias', 'roberta.encoder.layer.7.intermediate.dense.scale', 'roberta.encoder.layer.7.intermediate.dense.zero_point', 'roberta.encoder.layer.7.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.7.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.7.output.dense.scale', 'roberta.encoder.layer.7.output.dense.zero_point', 'roberta.encoder.layer.7.output.dense._packed_params.dtype', 'roberta.encoder.layer.7.output.dense._packed_params._packed_params', 'roberta.encoder.layer.7.output.LayerNorm.weight', 'roberta.encoder.layer.7.output.LayerNorm.bias', 'roberta.encoder.layer.8.attention.self.query.scale', 'roberta.encoder.layer.8.attention.self.query.zero_point', 'roberta.encoder.layer.8.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.8.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.8.attention.self.key.scale', 'roberta.encoder.layer.8.attention.self.key.zero_point', 'roberta.encoder.layer.8.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.8.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.8.attention.self.value.scale', 'roberta.encoder.layer.8.attention.self.value.zero_point', 'roberta.encoder.layer.8.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.8.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.8.attention.output.dense.scale', 'roberta.encoder.layer.8.attention.output.dense.zero_point', 'roberta.encoder.layer.8.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.8.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.8.attention.output.LayerNorm.weight', 'roberta.encoder.layer.8.attention.output.LayerNorm.bias', 'roberta.encoder.layer.8.intermediate.dense.scale', 'roberta.encoder.layer.8.intermediate.dense.zero_point', 'roberta.encoder.layer.8.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.8.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.8.output.dense.scale', 'roberta.encoder.layer.8.output.dense.zero_point', 'roberta.encoder.layer.8.output.dense._packed_params.dtype', 'roberta.encoder.layer.8.output.dense._packed_params._packed_params', 'roberta.encoder.layer.8.output.LayerNorm.weight', 'roberta.encoder.layer.8.output.LayerNorm.bias', 'roberta.encoder.layer.9.attention.self.query.scale', 'roberta.encoder.layer.9.attention.self.query.zero_point', 'roberta.encoder.layer.9.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.9.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.9.attention.self.key.scale', 'roberta.encoder.layer.9.attention.self.key.zero_point', 'roberta.encoder.layer.9.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.9.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.9.attention.self.value.scale', 'roberta.encoder.layer.9.attention.self.value.zero_point', 'roberta.encoder.layer.9.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.9.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.9.attention.output.dense.scale', 'roberta.encoder.layer.9.attention.output.dense.zero_point', 'roberta.encoder.layer.9.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.9.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.9.attention.output.LayerNorm.weight', 'roberta.encoder.layer.9.attention.output.LayerNorm.bias', 'roberta.encoder.layer.9.intermediate.dense.scale', 'roberta.encoder.layer.9.intermediate.dense.zero_point', 'roberta.encoder.layer.9.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.9.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.9.output.dense.scale', 'roberta.encoder.layer.9.output.dense.zero_point', 'roberta.encoder.layer.9.output.dense._packed_params.dtype', 'roberta.encoder.layer.9.output.dense._packed_params._packed_params', 'roberta.encoder.layer.9.output.LayerNorm.weight', 'roberta.encoder.layer.9.output.LayerNorm.bias', 'roberta.encoder.layer.10.attention.self.query.scale', 'roberta.encoder.layer.10.attention.self.query.zero_point', 'roberta.encoder.layer.10.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.10.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.10.attention.self.key.scale', 'roberta.encoder.layer.10.attention.self.key.zero_point', 'roberta.encoder.layer.10.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.10.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.10.attention.self.value.scale', 'roberta.encoder.layer.10.attention.self.value.zero_point', 'roberta.encoder.layer.10.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.10.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.10.attention.output.dense.scale', 'roberta.encoder.layer.10.attention.output.dense.zero_point', 'roberta.encoder.layer.10.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.10.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.10.attention.output.LayerNorm.weight', 'roberta.encoder.layer.10.attention.output.LayerNorm.bias', 'roberta.encoder.layer.10.intermediate.dense.scale', 'roberta.encoder.layer.10.intermediate.dense.zero_point', 'roberta.encoder.layer.10.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.10.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.10.output.dense.scale', 'roberta.encoder.layer.10.output.dense.zero_point', 'roberta.encoder.layer.10.output.dense._packed_params.dtype', 'roberta.encoder.layer.10.output.dense._packed_params._packed_params', 'roberta.encoder.layer.10.output.LayerNorm.weight', 'roberta.encoder.layer.10.output.LayerNorm.bias', 'roberta.encoder.layer.11.attention.self.query.scale', 'roberta.encoder.layer.11.attention.self.query.zero_point', 'roberta.encoder.layer.11.attention.self.query._packed_params.dtype', 'roberta.encoder.layer.11.attention.self.query._packed_params._packed_params', 'roberta.encoder.layer.11.attention.self.key.scale', 'roberta.encoder.layer.11.attention.self.key.zero_point', 'roberta.encoder.layer.11.attention.self.key._packed_params.dtype', 'roberta.encoder.layer.11.attention.self.key._packed_params._packed_params', 'roberta.encoder.layer.11.attention.self.value.scale', 'roberta.encoder.layer.11.attention.self.value.zero_point', 'roberta.encoder.layer.11.attention.self.value._packed_params.dtype', 'roberta.encoder.layer.11.attention.self.value._packed_params._packed_params', 'roberta.encoder.layer.11.attention.output.dense.scale', 'roberta.encoder.layer.11.attention.output.dense.zero_point', 'roberta.encoder.layer.11.attention.output.dense._packed_params.dtype', 'roberta.encoder.layer.11.attention.output.dense._packed_params._packed_params', 'roberta.encoder.layer.11.attention.output.LayerNorm.weight', 'roberta.encoder.layer.11.attention.output.LayerNorm.bias', 'roberta.encoder.layer.11.intermediate.dense.scale', 'roberta.encoder.layer.11.intermediate.dense.zero_point', 'roberta.encoder.layer.11.intermediate.dense._packed_params.dtype', 'roberta.encoder.layer.11.intermediate.dense._packed_params._packed_params', 'roberta.encoder.layer.11.output.dense.scale', 'roberta.encoder.layer.11.output.dense.zero_point', 'roberta.encoder.layer.11.output.dense._packed_params.dtype', 'roberta.encoder.layer.11.output.dense._packed_params._packed_params', 'roberta.encoder.layer.11.output.LayerNorm.weight', 'roberta.encoder.layer.11.output.LayerNorm.bias', 'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'fc.scale', 'fc.zero_point', 'fc._packed_params.dtype', 'fc._packed_params._packed_params']
# Select and save the specified attributes
for attribute_name in attributes_to_save:
    # print(attribute_name)
    selected_attributes[attribute_name] = state_dict[attribute_name]

# Save the selected attributes to a file
# print(selected_attributes)

# for name, param in state_dict.items():
#     print(f"Layer name: {name}, Size: {param}")

torch.save(selected_attributes, "classifier_attributes.pth")
print('torch saved')

# Evaluation on the validation set
lstm_classifier.eval()
val_preds = []
val_true = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = lstm_classifier(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(labels.cpu().numpy())

# Convert predictions and ground truth to their original labels
pred_labels = [label for label, index in label_mapping.items() for pred in val_preds if index == pred]
true_labels = [label for label, index in label_mapping.items() for true in val_true if index == true]

# Print classification report
print("test data")
print(accuracy_score(true_labels, pred_labels))
print(classification_report(true_labels, pred_labels))





