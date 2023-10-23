import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report
from custom_classifier import SequenceClassifier


train_data_path = '../data/train.json'
validation_data_path = '../data/validation.json'
test_data_path = '../data/test.json'

# Loading the training, validation, and test data from JSON files
def load_data(data_path):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    return data

train_data = load_data(train_data_path)

test_data = load_data(test_data_path)


label_mapping = {"Churn": 0, "Escalation": 1, 'Churn and Escalation':2, "No Intent Found": 3,}


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


#Using the pretrained bert models for tokenising and embedding creation.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))

train_input_ids, train_attention_mask, train_labels = preprocess_data(train_data, tokenizer, label_mapping)
test_input_ids, test_attention_mask, test_labels = preprocess_data(test_data, tokenizer, label_mapping)


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


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


num_labels = len(label_mapping)

custom_classifier = SequenceClassifier(input_dim=bert_model.config.hidden_size, hidden_dim_1=264, hidden_dim_2= 64, num_layers =1, num_classes=len(label_mapping))

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_classifier.parameters(), lr=1e-3) 

# Training custom classifier model
for epoch in range(20):
    custom_classifier.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        optimizer.zero_grad()
        
        # Get BERT embeddings
        with torch.no_grad():
            bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            bert_embeddings = bert_outputs.last_hidden_state
        
        # Pass BERT embeddings through custom classifier
        custom_outputs = custom_classifier(bert_embeddings)
        
        loss = criterion(custom_outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss: {total_loss / len(train_loader)}")
torch.save(custom_classifier, '../data/custom.pth')

# Evaluation on the test set
test_true_labels = []
test_pred_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        with torch.no_grad():
                bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                bert_embeddings = bert_outputs.last_hidden_state
                

        outputs = custom_classifier(bert_embeddings)
        logits = outputs

        predicted_labels = torch.argmax(logits, dim=1).tolist()
        test_true_labels.extend(labels.tolist())
        test_pred_labels.extend(predicted_labels)

test_accuracy = accuracy_score(test_true_labels, test_pred_labels)
test_classification_report = classification_report(test_true_labels, test_pred_labels)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:")
print(test_classification_report)

train_true_labels = []
train_pred_labels = []

with torch.no_grad():
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        with torch.no_grad():
            bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            bert_embeddings = bert_outputs.last_hidden_state
        

        outputs = custom_classifier(bert_embeddings)
        logits = outputs

        predicted_labels = torch.argmax(logits, dim=1).tolist()
        train_true_labels.extend(labels.tolist())
        train_pred_labels.extend(predicted_labels)

train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
train_classification_report = classification_report(train_true_labels, train_pred_labels)

print(f"Train Accuracy: {train_accuracy:.4f}")
print("Train Classification Report:")
print(train_classification_report)



