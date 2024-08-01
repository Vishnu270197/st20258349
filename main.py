import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Function to read CSV dataset
def read_csv_data(file_path):
    data = pd.read_csv(file_path)
    sentences = []
    sentence = []
    for word, tag in zip(data['word'], data['tag']):
        if pd.isna(word):
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append((word, tag))
    if sentence:
        sentences.append(sentence)
    return sentences


# Read data from CSV files
train_data = read_csv_data("datasets/train.csv")
valid_data = read_csv_data("datasets/valid.csv")
test_data = read_csv_data("datasets/test.csv")
print("datasets are loaded from respective csv files")

# Define a label map
label_map = {
    "O": 0,
    "B-ORG": 1,
    "I-ORG": 2,
    "B-MISC": 3,
    "I-MISC": 4,
    "B-PER": 5,
    "I-PER": 6,
    "B-LOC": 7,
    "I-LOC": 8
}

# Tokenization and dataset preparation
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


def preprocess_data(data, tokenizer, max_len=128):
    input_ids, attention_masks, labels = [], [], []

    for sent in data:
        words, tags = zip(*sent)
        encodings = tokenizer(list(words),
                              is_split_into_words=True,
                              return_tensors="pt",
                              padding='max_length',
                              truncation=True,
                              max_length=max_len)
        input_ids.append(encodings['input_ids'])
        attention_masks.append(encodings['attention_mask'])
        label_ids = [label_map.get(tag, 0) for tag in tags
                     ] + [0] * (max_len - len(tags))  # Pad labels
        labels.append(
            torch.tensor(label_ids).unsqueeze(0))  # Add batch dimension

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    labels = torch.cat(labels)
    return TensorDataset(input_ids, attention_masks, labels)


train_dataset = preprocess_data(train_data, tokenizer)
valid_dataset = preprocess_data(valid_data, tokenizer)
test_dataset = preprocess_data(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Model definition
model = BertForTokenClassification.from_pretrained('bert-base-cased',
                                                   num_labels=len(label_map))

# Training setup
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 10
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


# Training loop with visualization
def train_model(model, train_loader, valid_loader, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            b_input_ids, b_attention_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids,
                            attention_mask=b_attention_mask,
                            labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_train_loss}")

        # Validation
        model.eval()
        eval_loss = 0
        for batch in valid_loader:
            b_input_ids, b_attention_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids,
                                attention_mask=b_attention_mask,
                                labels=b_labels)
            loss = outputs.loss
            eval_loss += loss.item()

        avg_val_loss = eval_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1),
             train_losses,
             label="Training Loss",
             marker='o')
    plt.plot(range(1, epochs + 1),
             val_losses,
             label="Validation Loss",
             marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.show()


train_model(model, train_loader, valid_loader)


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []

    for batch in test_loader:
        b_input_ids, b_attention_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=2).flatten().tolist())
        true_labels.extend(b_labels.flatten().tolist())

    # Filter out padding
    filtered_preds, filtered_labels = [], []
    for pred, label in zip(predictions, true_labels):
        if label != 0:  # Ignore padding
            filtered_preds.append(pred)
            filtered_labels.append(label)

    unique_preds = set(filtered_preds)
    unique_labels = set(filtered_labels)

    print(f"Unique predicted classes: {unique_preds}")
    print(f"Unique true label classes: {unique_labels}")

    target_names = list(label_map.keys())[1:]  # Exclude 'O' from target names
    report = classification_report(filtered_labels,
                                   filtered_preds,
                                   target_names=target_names,
                                   labels=list(unique_labels))
    print(report)


# Call the evaluation function
evaluate_model(model, test_loader)
