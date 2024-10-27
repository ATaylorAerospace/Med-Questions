import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Define the training and validation datasets
train_dataset = ...  # Add your training dataset here
val_dataset = ...  # Add your validation dataset here

# Define the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the training parameters
batch_size = 4  # Adjust the batch size as needed
num_epochs = 5  # Adjust the number of training epochs as needed
learning_rate = 2e-5  # Adjust the learning rate as needed

# Define the DataLoader
def collate_fn(batch):
    input_ids, attention_masks, token_type_ids, labels = zip(*batch)
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'token_type_ids': torch.stack(token_type_ids),
        'labels': torch.tensor(labels)
    }

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define the training optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs

# Define the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define the training function
def train(model, optimizer, scheduler, train_loader):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            token_type_ids = batch['token_type_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Define the validation function
def validate(model, val_loader):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            token_type_ids = batch['token_type_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            predicted_labels = torch.argmax(outputs.logits, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions, average='macro')
    recall = recall_score(actual_labels, predictions, average='macro')
    precision = precision_score(actual_labels, predictions, average='macro')
    return accuracy, f1, recall, precision

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
train(model, optimizer, scheduler, train_loader)

# Evaluate the model on the validation dataset
accuracy, f1, recall, precision = validate(model, val_loader)
print('Validation accuracy:', accuracy)
print('Validation f1:', f1)
print('Validation recall:', recall)
print('Validation precision:', precision)

# Save the trained model
model_path = 'trained_model.pt'
torch.save(model.state_dict(), model_path)
print('Trained model saved to:', model_path)
